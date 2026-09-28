"""Short, readable summaries of what a node was given and returned.

A trace is read by a person looking at a timeline, so the summaries name the
useful fields and stay short. They also drop anything sensitive: a trace should
not become a second copy of the conversation.
"""

from __future__ import annotations

from typing import Any

MAX_CHARS = 600

# Fields worth naming in a trace, in the order they are shown.
INTERESTING_INPUTS = (
    "user_input",
    "intent",
    "intent_conf",
    "urgency_band",
    "frustrated",
    "needs_human",
    "escalate",
    "retrieval_query",
    "pending_action",
    "approval",
    "ticket_id",
)

INTERESTING_OUTPUTS = (
    "intent",
    "intent_conf",
    "response",
    "pending_action",
    "approval",
    "ticket_id",
    "tools_used",
    "escalate",
    "memory_updated",
    "llm_note",
)

# Never written into a trace, whatever else is included.
REDACTED = {"messages", "router_state", "retrieved_context", "customer_memory", "raw"}


def _render(value: Any) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return ""
    text = str(value).replace("\n", " ")
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "..."


def summarise(state: dict[str, Any], fields: tuple[str, ...]) -> str:
    """`field=value` pairs for the interesting fields that are set."""
    parts = []
    for name in fields:
        if name in REDACTED:
            continue
        rendered = _render(state.get(name))
        if rendered:
            parts.append(f"{name}={rendered}")
    return " ".join(parts)


def summarise_input(state: dict[str, Any]) -> str:
    """What the node was given."""
    return summarise(state, INTERESTING_INPUTS) or "(empty state)"


def summarise_output(result: dict[str, Any]) -> str:
    """What the node returned."""
    return summarise(result, INTERESTING_OUTPUTS) or "(no output)"
