"""The LangGraph state for one conversation thread.

Field names `user_input`, `retrieved_context` and `escalate` are kept from the
original agent so the shape stays recognisable. `messages` uses `add_messages`,
which appends and merges by id, and `trace` accumulates with `operator.add`, so
a node returning one entry does not erase what earlier nodes recorded.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

Approval = Literal["approved", "rejected", None]


class SupportState(TypedDict, total=False):
    """Everything one turn of the support conversation needs.

    Every field is optional: the graph fills them as it goes, and a resumed
    thread continues from the checkpoint rather than starting over.
    """

    # --- conversation ------------------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    customer_id: str | None
    thread_id: str
    # A short stand-in for the user's message used for retrieval, written by the
    # faq agent when the customer greets first and asks later.
    retrieval_query: str | None

    # --- router decisions -------------------------------------------------
    intent: str | None
    intent_conf: float | None
    intent_probs: dict[str, float]
    urgency: float | None
    urgency_band: str | None
    frustrated: bool | None
    # P(injection) on 0..1, not a bool: the guardrail has a threshold.
    injection: float | None
    needs_human: bool | None
    # Which signal decided the escalation: a keyword, a request, a topic.
    needs_human_reason: str | None
    # Kept from the original agent; now derived from the router rather than a
    # keyword list.
    escalate: bool

    # --- RAG --------------------------------------------------------------
    retrieved_context: str
    retrieved: list[dict[str, Any]]

    # --- actions ----------------------------------------------------------
    # e.g. {"type": "refund", "invoice_id": ..., "amount": 19.0, "mode": "staff_approve"}
    pending_action: dict[str, Any] | None
    approval: Approval
    # The note a reviewer left with their decision.
    approval_note: str | None
    # Who approved it, for the refund audit trail.
    approved_by: str | None
    # "staff_approve" or "customer_confirm", set by the gate.
    mode: str | None
    # What the action actually did, once run.
    action_result: dict[str, Any] | None
    # The interrupt payload the API hands to the client.
    interrupt: dict[str, Any] | None
    ticket_id: str | None
    # The full ticket row, so `escalate` can quote the id.
    ticket: dict[str, Any] | None
    response: str

    # --- memory -----------------------------------------------------------
    customer_memory: str | None
    # Whether the long-term summary was rewritten this turn.
    memory_updated: bool
    # Name, plan, devices and memory, loaded once per turn.
    customer_facts: dict[str, Any]
    # Set when the router was unsure and a clarifying question was asked.
    awaiting_clarification: bool
    # Set by the guardrail when the turn looked like an injection attempt.
    unsafe: bool
    # Set by an agent that wants a ticket opened before it finishes.
    wants_ticket: bool
    # The router's answers as plain data, for the trace and the UI. Not the
    # RouterDecision object: state is written to the checkpointer.
    router_decision: dict[str, Any]
    # The exact text the router saw, kept so a decision can be explained.
    router_state: str

    # --- observability ----------------------------------------------------
    trace: Annotated[list[dict[str, Any]], operator.add]
    run_id: str
    # Which tools ran this turn, for the UI chips.
    tools_used: list[str]
    # Set when the LLM layer had to fall back or give up.
    llm_note: str | None
    escalated_reason: str | None


def new_state(thread_id: str, customer_id: str | None, user_input: str) -> SupportState:
    """The starting state for a fresh turn."""
    return {
        "messages": [],
        "user_input": user_input,
        "customer_id": customer_id,
        "thread_id": thread_id,
        "retrieved_context": "",
        "retrieved": [],
        "escalate": False,
        "pending_action": None,
        "approval": None,
        "interrupt": None,
        "ticket_id": None,
        "response": "",
        "customer_memory": None,
        "awaiting_clarification": False,
        "trace": [],
        "tools_used": [],
    }


def last_customer_message(messages: list[AnyMessage]) -> str:
    """The text of the most recent customer message, or an empty string."""
    from langchain_core.messages import HumanMessage

    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def conversation_turns(messages: list[AnyMessage], limit: int = 2) -> list[tuple[str, str]]:
    """The last `limit` turns as `(role, text)`, oldest first, for the router.

    The router state has a 512-token budget, so this is capped hard: the newest
    turn is the one that matters and older context is dropped first.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    turns: list[tuple[str, str]] = []
    for message in messages or []:
        if isinstance(message, HumanMessage):
            turns.append(("Customer", str(message.content)))
        elif isinstance(message, AIMessage):
            turns.append(("Agent", str(message.content)))
    return turns[-limit:] if limit > 0 else []
