"""Conditional edge functions for the support graph.

These are deliberately pure: they read the state and return a node name, with no
LLM call, no database and no side effect. That is what makes the routing
testable in isolation and what replaced the original's
`route_after_escalation_check` stub that always returned "generate".
"""

from __future__ import annotations

from typing import Any, Literal

from ..config import Settings, get_settings
from ..models.router import INJECTION_THRESHOLD

# Which node handles which intent.
BILLING_INTENTS = frozenset({"billing", "refund", "cancellation"})
FAQ_INTENTS = frozenset({"product_info", "pricing", "account", "smalltalk"})
TECHNICAL_INTENTS = frozenset({"technical"})

# "High enough that an angry customer needs a person", as a 0..1 urgency.
ANGRY_URGENCY = 0.5


def _get(state: dict[str, Any], key: str, default: Any = None) -> Any:
    value = state.get(key, default)
    return default if value is None else value


def needs_human(state: dict[str, Any]) -> bool:
    """True when a person has to take this turn, whatever the intent.

    Covers the three ways a turn escalates: the router decided it, the customer
    is angry *and* it is urgent, or the model itself asked to be handed over.
    """
    if _get(state, "needs_human", False):
        return True
    if _get(state, "escalate", False):
        return True
    if _get(state, "frustrated", False) and (_get(state, "urgency", 0.0) or 0.0) >= ANGRY_URGENCY:
        return True
    if state.get("escalated_reason") == "agent_requested":
        return True
    return False


def is_low_confidence(state: dict[str, Any], settings: Settings | None = None) -> bool:
    """True when the router was not sure enough to act.

    The first low-confidence turn asks a clarifying question; if the next turn is
    still unsure, the graph escalates rather than guessing a second time.
    """
    settings = settings or get_settings()
    confidence = _get(state, "intent_conf", 1.0)
    return confidence < settings.router_min_conf


def route_by_intent(state: dict[str, Any], settings: Settings | None = None) -> str:
    """Where to send this turn after the guardrail.

    Args:
        state: the graph state.
        settings: for `ROUTER_MIN_CONF`.

    Returns:
        A node name: "escalate", "clarify", or one of the agent nodes.
    """
    if is_injection(state):
        return "respond"

    if needs_human(state):
        return "escalate"

    if is_low_confidence(state, settings):
        # One clarifying question. If the customer is still unclear on the next
        # turn, `awaiting_clarification` is set and we stop guessing.
        if state.get("awaiting_clarification"):
            return "escalate"
        return "clarify"

    intent = _get(state, "intent", "product_info")
    if intent in BILLING_INTENTS:
        return "billing_agent"
    if intent in TECHNICAL_INTENTS:
        return "technical_agent"
    if intent in FAQ_INTENTS:
        return "faq_agent"
    # An intent this graph does not know is treated as a general question rather
    # than sent somewhere arbitrary.
    return "faq_agent"


def is_injection(state: dict[str, Any]) -> bool:
    """True when the turn looks like an attempt to override the assistant."""
    injection = _get(state, "injection", 0.0)
    if isinstance(injection, bool):
        return injection
    return float(injection or 0.0) > INJECTION_THRESHOLD


def after_agent(state: dict[str, Any]) -> str:
    """Where to go after an agent node has produced its answer.

    An agent that could not answer from the knowledge base offers a ticket, and
    saying yes means creating one; otherwise the turn ends at `respond`.
    """
    if state.get("ticket_id"):
        return "create_ticket"
    if _get(state, "wants_ticket", False):
        return "create_ticket"
    return "respond"


def after_billing(state: dict[str, Any]) -> str:
    """Where to go after the billing agent.

    A proposed action pauses for approval; without one the turn simply answers.
    """
    if state.get("pending_action"):
        return "approval_gate"
    return "respond"


def after_approval(state: dict[str, Any]) -> str:
    """Run the action only if it was approved; otherwise just answer."""
    if _get(state, "approval", None) == "approved" and state.get("pending_action"):
        return "execute_action"
    return "respond"
