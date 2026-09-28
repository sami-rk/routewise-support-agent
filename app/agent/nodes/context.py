"""Nodes that run before routing: load the customer's context, route, guard.

`load_context` reads the profile, subscription and long-term memory so the agent
prompts already know who they are talking to, and trims the message history to
`HISTORY_MAX_MESSAGES` so a long thread cannot grow the prompt without bound.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import trim_messages

from ...config import Settings, get_settings
from ...memory.long_term import customer_facts
from ...models.factory import get_router
from ..state import SupportState, conversation_turns, last_customer_message


def load_context(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Load the customer's profile, subscription and memory; trim the history.

    Reads no LLM and writes nothing, so it costs no request on the free tier.
    """
    settings = settings or get_settings()
    customer_id = state.get("customer_id")
    facts = customer_facts(customer_id) if customer_id else {}

    messages = list(state.get("messages") or [])
    # trim_messages counts tokens; the token counter is local, so this is free.
    trimmed = trim_messages(
        messages,
        max_tokens=settings.history_max_messages,
        token_counter=len,
        strategy="last",
        include_system=False,
        start_on="human",
    )

    return {
        "customer_memory": facts.get("memory"),
        "customer_facts": facts,
        "messages": trimmed,
    }


def laya_router(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Route the turn with the local decision model. No OpenRouter request.

    The state sent to the router is the customer's latest message plus at most
    two earlier turns and a one-line customer summary, because the checkpoint
    has a 512-token context.
    """
    from ...models.router import build_router_state

    messages = list(state.get("messages") or [])
    user_input = state.get("user_input") or last_customer_message(messages)

    facts = state.get("customer_facts") or {}
    summary = describe_customer(facts)
    router_state = build_router_state(user_input, conversation_turns(messages[:-1] or messages, 2), summary)

    decision = get_router().predict(router_state)

    return {
        "intent": decision.intent,
        "intent_conf": decision.intent_conf,
        "intent_probs": decision.intent_probs,
        "urgency": decision.urgency,
        "urgency_band": decision.urgency_band,
        "frustrated": decision.frustrated,
        "needs_human": decision.needs_human,
        "needs_human_reason": decision.needs_human_reason,
        "injection": decision.injection_conf or 0.0,
        "router_decision": decision,
        "router_state": router_state,
        # Kept from the original agent: now derived from the router, not keywords.
        "escalate": decision.needs_human,
        "retrieval_query": user_input,
    }


def describe_customer(facts: dict[str, Any] | None) -> str:
    """One short line about the customer, for the router's state.

    Kept deliberately short: it shares the 512-token budget with the message.
    """
    facts = facts or {}
    parts = []
    if facts.get("plan"):
        parts.append(f"{facts['plan']} plan")
    if facts.get("devices"):
        parts.append(f"{facts['devices']} devices")
    if facts.get("memory"):
        parts.append(str(facts["memory"])[:120])
    return ", ".join(parts)


# What the guardrail says when it stops a message. The customer is told plainly
# that the message was not acted on, without any hint about the system.
INJECTION_REPLY = (
    "I can only help with CloudSync Pro questions, so I could not act on that "
    "message. What do you need help with?"
)


def guardrail(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Stop prompt injection before it reaches an agent or a tool.

    Short-circuits to `respond` with a fixed, safe reply. Nothing from the
    message is passed on, so there is nothing for it to influence.
    """
    from ..routing import is_injection

    if is_injection(dict(state)):
        return {
            "response": INJECTION_REPLY,
            "unsafe": True,
            "escalate": False,
        }
    return {"unsafe": False}
