"""Build and compile the support graph.

    START -> load_context -> laya_router -> guardrail
    guardrail --(injection)--> respond
    guardrail --> route_by_intent
    route_by_intent --> clarify | escalate | retrieve_kb
    retrieve_kb --> billing_agent | technical_agent | faq_agent
    billing_agent --> (proposed action) approval_gate --> execute_action --> respond
    billing_agent --> (no action) respond
    *_agent --> (ticket wanted) create_ticket --> respond
    escalate --> create_ticket --> respond
    respond --> update_memory --> END

Compiled with a `SqliteSaver`, so a thread survives a restart and a turn
interrupted for approval resumes exactly where it stopped.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ..config import Settings, get_settings
from .nodes.actions import (
    approval_gate,
    create_ticket_node,
    escalate,
    execute_action,
    respond,
)
from .nodes.agents import billing_agent, faq_agent, technical_agent
from .nodes.context import guardrail, laya_router, load_context
from .nodes.memory_node import update_memory
from .nodes.retrieval import retrieve_kb
from .routing import after_agent, after_approval, after_billing, route_by_intent
from .state import SupportState


def clarify(state: SupportState) -> dict[str, Any]:
    """Ask one question when the router was not confident.

    Costs one LLM call, but only on the turn where the router was unsure, and it
    is what stops a vague message being sent to the wrong agent.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from ..models.llm import LLMUnavailableError
    from ..models.llm_runner import invoke_with_fallback
    from . import prompts
    from .nodes.shared import message_text

    question = (
        "Happy to help — is this about your plan or invoices, a problem with the app, "
        "or your account settings?"
    )
    try:
        response = invoke_with_fallback(
            [
                SystemMessage(prompts.clarify_prompt(state)),
                HumanMessage(state.get("user_input") or ""),
            ],
            role="clarify",
            run_id=state.get("run_id"),
            thread_id=state.get("thread_id"),
        )
        question = message_text(response) or question
    except LLMUnavailableError:
        # Even when the model is unavailable, the fixed question is a better
        # answer than a wrong route.
        pass

    return {
        "response": question,
        "awaiting_clarification": True,
        "messages": [AIMessage(content=question)],
    }


def build_graph(checkpointer: Any = None, settings: Settings | None = None):
    """Build the StateGraph and wire every edge.

    Args:
        checkpointer: a saver to persist threads with. Without one the graph runs
            but remembers nothing between turns and cannot pause for approval.
        settings: for `ROUTER_MIN_CONF` and `REFUND_AUTO_LIMIT`.

    Returns:
        The compiled graph, ready for `invoke` or `stream`.
    """
    settings = settings or get_settings()

    graph = StateGraph(SupportState)

    graph.add_node("load_context", load_context)
    graph.add_node("laya_router", laya_router)
    graph.add_node("guardrail", guardrail)
    graph.add_node("retrieve_kb", retrieve_kb)
    graph.add_node("faq_agent", faq_agent)
    graph.add_node("billing_agent", billing_agent)
    graph.add_node("technical_agent", technical_agent)
    graph.add_node("clarify", clarify)
    graph.add_node("approval_gate", approval_gate)
    graph.add_node("execute_action", execute_action)
    graph.add_node("create_ticket", create_ticket_node)
    graph.add_node("escalate", escalate)
    graph.add_node("respond", respond)
    graph.add_node("update_memory", update_memory)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "laya_router")
    graph.add_edge("laya_router", "guardrail")

    # The guardrail short-circuits an injection attempt straight to the reply;
    # otherwise the intent decides. `route_by_intent` is a pure function rather
    # than a node, so the two live in one conditional edge here.
    graph.add_conditional_edges(
        "guardrail",
        lambda state: "respond" if state.get("unsafe") else route_by_intent(state, settings),
        {
            "clarify": "clarify",
            "escalate": "escalate",
            # All three agents go through retrieval first, so they have passages.
            "billing_agent": "retrieve_kb",
            "technical_agent": "retrieve_kb",
            "faq_agent": "retrieve_kb",
            "retrieve_kb": "retrieve_kb",
            "respond": "respond",
        },
    )

    # After retrieval, the same decision picks the specialist.
    graph.add_conditional_edges(
        "retrieve_kb",
        lambda state: route_by_intent(state, settings),
        {
            "billing_agent": "billing_agent",
            "technical_agent": "technical_agent",
            "faq_agent": "faq_agent",
            "escalate": "escalate",
            "clarify": "clarify",
            "respond": "respond",
        },
    )

    # clarify and escalate both end the turn; escalate first opens a ticket.
    graph.add_edge("clarify", "respond")
    graph.add_edge("escalate", "create_ticket")

    graph.add_conditional_edges(
        "billing_agent",
        after_billing,
        {"approval_gate": "approval_gate", "respond": "respond"},
    )
    graph.add_conditional_edges(
        "approval_gate",
        after_approval,
        {"execute_action": "execute_action", "respond": "respond"},
    )
    graph.add_edge("execute_action", "respond")

    for agent in ("faq_agent", "technical_agent"):
        graph.add_conditional_edges(
            agent,
            after_agent,
            {"create_ticket": "create_ticket", "respond": "respond"},
        )

    graph.add_edge("create_ticket", "respond")
    graph.add_edge("respond", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile(checkpointer=checkpointer)


@contextmanager
def persistent_graph(settings: Settings | None = None) -> Iterator[Any]:
    """A compiled graph whose threads are persisted in SQLite.

    `SqliteSaver.from_conn_string` is a context manager, so the connection has to
    outlive the graph. Use this in the FastAPI lifespan and the CLI:

        with persistent_graph() as graph:
            graph.invoke(state, config)
    """
    settings = settings or get_settings()
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as saver:
        yield build_graph(checkpointer=saver, settings=settings)


def thread_config(thread_id: str) -> dict[str, Any]:
    """The config that pins a run to one conversation thread."""
    return {"configurable": {"thread_id": thread_id}}
