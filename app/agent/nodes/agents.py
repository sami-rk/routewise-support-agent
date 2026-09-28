"""The three specialist agents, and the tool loop they share.

Each agent makes one LLM call, calls a tool if the model asks, and calls again
with the tool result. The loop is bounded by `MAX_TOOL_STEPS` because the free
tier allows about 20 requests a minute, and a model that keeps asking for tools
must not be able to spend a turn's whole budget.

When retrieval found nothing, the agent is given a prompt that tells it to say it
does not know and offer a ticket, rather than letting a weak model improvise an
answer from an empty context.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from langchain_core.messages import SystemMessage

from ...config import Settings, get_settings
from ...models.llm import BUSY_MESSAGE, LLMUnavailableError
from ...models.llm_runner import invoke_with_fallback
from ...tools.registry import ToolError, tools_for_agent
from .. import prompts
from ..state import SupportState
from .retrieval import NOTHING_FOUND, has_context
from .shared import friendly_error, message_text, tool_result_message

logger = logging.getLogger(__name__)

# How many tool round trips one agent turn may take. Two keeps a turn at three
# LLM calls, which the free tier can absorb.
MAX_TOOL_STEPS = 2

# An agent proposes a money or cancellation action with a line like this.
ACTION_RE = re.compile(r"ACTION:\s*(\{.*?\})\s*$", re.IGNORECASE | re.DOTALL | re.MULTILINE)


def trim_for_prompt(messages: list[Any], keep: int = 8) -> list[Any]:
    """The most recent exchanges, so a long thread stays cheap to answer."""
    return messages[-keep:] if len(messages) > keep else messages


def run_agent(
    state: SupportState,
    *,
    role: str,
    agent: str,
    system_prompt: Callable[[SupportState], str],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """One agent turn: call the model, let it use tools, then answer.

    Args:
        state: the graph state.
        role: the LLM role, for logging and metrics.
        agent: which tool whitelist to use.
        system_prompt: builds the system prompt from the state.
        settings: settings to use.

    Returns:
        Partial state with `response`, `tools_used`, and `pending_action` when
        the model proposed one.
    """
    settings = settings or get_settings()

    if not has_context(state):
        # Nothing above the similarity floor: say so instead of guessing.
        prompt = prompts.no_answer_prompt(state)
    else:
        prompt = system_prompt(state)

    customer_id = state.get("customer_id")
    try:
        tools = tools_for_agent(customer_id, agent)
    except ToolError as exc:
        return {"response": str(exc), "tools_used": [], "pending_action": None}

    tool_names = [t.name for t in tools]
    tools_used: list[str] = []
    observations: list[Any] = []
    answer = ""

    for step in range(MAX_TOOL_STEPS + 1):
        # The last pass is unbound: a model that spends every turn calling tools
        # would otherwise end the turn with an empty message, and the customer
        # would get the generic "I am not sure" fallback instead of an answer.
        bind_tools = tools if step < MAX_TOOL_STEPS or not tools else None
        try:
            response = invoke_with_fallback(
                [SystemMessage(prompt), *_prompt_messages(state, observations)],
                role=role,
                settings=settings,
                tools=bind_tools or None,
                run_id=state.get("run_id"),
                thread_id=state.get("thread_id"),
            )
        except LLMUnavailableError as exc:
            return {
                "response": str(exc) or BUSY_MESSAGE,
                "tools_used": tools_used,
                "pending_action": None,
                "llm_note": "rate_limited",
            }

        answer = message_text(response)

        calls = getattr(response, "tool_calls", None) or []
        if not calls or not bind_tools:
            break

        observations = _run_calls(calls, tools, tool_names, tools_used)

    action = _proposed_action(answer, observations)
    return {
        "response": strip_action_line(answer),
        "tools_used": tools_used,
        "pending_action": action,
        "wants_ticket": bool(action and action.get("type") == "ticket"),
    }


def _run_calls(
    calls: list[dict[str, Any]],
    tools: list[Any],
    tool_names: list[str],
    tools_used: list[str],
) -> list[Any]:
    """Run each requested tool and collect the results for the next call."""
    observations: list[Any] = []
    for call in calls:
        name = call.get("name")
        if name not in tool_names:
            observations.append(tool_result_message(call, f"There is no tool called {name!r}."))
            continue
        tools_used.append(name)
        try:
            observations.append(tool_result_message(call, _run_tool(tools, name, call.get("args") or {})))
        except ToolError as exc:
            observations.append(tool_result_message(call, str(exc)))
        except Exception as exc:  # noqa: BLE001 - a tool failure must not end the turn
            logger.warning("tool %s failed: %s", name, exc)
            observations.append(
                tool_result_message(call, "That lookup did not work. Say so honestly.")
            )
    observations.append(
        "Now answer the customer in plain words. Do not call another tool unless you "
        "genuinely still need one."
    )
    return observations


def _proposed_action(answer: str, observations: list[Any]) -> dict[str, Any] | None:
    """The action the agent proposed, from a tool call or from the reply.

    A tool call is preferred: free models follow a tool schema far more reliably
    than a bare `ACTION: {...}` line, and the arguments arrive already parsed.
    The line format is still accepted, for models that manage it.
    """
    for observation in observations:
        name = getattr(observation, "name", None)
        if name == "propose_refund":
            arguments = _tool_arguments(observation)
            if arguments:
                return {
                    "type": "refund",
                    "invoice_id": arguments.get("invoice_id"),
                    "amount": arguments.get("amount"),
                    "reason": arguments.get("reason"),
                }
        if name == "propose_cancellation":
            return {"type": "cancel"}

    action = parse_action(answer)
    if action and action.get("type") == "cancel":
        return {"type": "cancel"}
    return action


def _tool_arguments(message: Any) -> dict[str, Any]:
    """The arguments of the tool call a `ToolMessage` is answering."""
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, dict):
        return artifact
    return {}


def _prompt_messages(state: SupportState, observations: list[Any]) -> list[Any]:
    """The conversation so far, plus any tool results from this turn."""
    from langchain_core.messages import AIMessage

    history = [m for m in (state.get("messages") or []) if not isinstance(m, AIMessage)]
    return [*trim_for_prompt(history), *observations]


def _run_tool(tools: list[Any], name: str, args: dict[str, Any]) -> str:
    for tool in tools:
        if tool.name == name:
            return str(tool.invoke(args))
    raise ToolError(f"There is no tool called {name!r}.")


def parse_action(text: str) -> dict[str, Any] | None:
    """Read the `ACTION: {...}` line a model may have added.

    Returns:
        The parsed action, or None. Anything unparseable is ignored rather than
        trusted: the gate re-reads the invoice before acting anyway.
    """
    import json

    match = ACTION_RE.search(text or "")
    if not match:
        return None
    try:
        action = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return action if isinstance(action, dict) and action.get("type") else None


def strip_action_line(text: str) -> str:
    """The reply without the machine-readable action line."""
    return ACTION_RE.sub("", text or "").strip()


# --- the three agents ------------------------------------------------------


def faq_agent(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Product, pricing, how-to and small talk, answered from the knowledge base."""
    return run_agent(
        state,
        role="faq",
        agent="faq",
        system_prompt=lambda s: prompts.faq_system_prompt(s.get("customer_facts") or {}, s),
        settings=settings,
    )


def billing_agent(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Subscriptions, invoices, refunds and cancellations, with tools."""
    return run_agent(
        state,
        role="billing",
        agent="billing",
        system_prompt=lambda s: prompts.billing_system_prompt(s.get("customer_facts") or {}, s),
        settings=settings,
    )


def technical_agent(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Sync and platform troubleshooting from the knowledge base."""
    return run_agent(
        state,
        role="technical",
        agent="technical",
        system_prompt=lambda s: prompts.technical_system_prompt(s.get("customer_facts") or {}, s),
        settings=settings,
    )
