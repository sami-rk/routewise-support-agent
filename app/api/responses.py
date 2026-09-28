"""Turn a graph result into an API response.

One place decides what a client sees, so `POST /chat`, the SSE stream and the CLI
cannot drift apart: in particular, an interrupted turn always looks the same
whichever way it was started.
"""

from __future__ import annotations

from typing import Any

from .models import ChatResponse

STATUS_FOR_MODE = {
    "staff_approve": "awaiting_approval",
    "customer_confirm": "awaiting_confirmation",
}


def interrupt_of(result: dict[str, Any]) -> dict[str, Any] | None:
    """The interrupt payload, whichever key LangGraph put it under."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"value": value}


def citations_of(result: dict[str, Any]) -> list[str]:
    """The knowledge base sources this turn used."""
    return [
        str(passage.get("citation"))
        for passage in (result.get("retrieved") or [])
        if passage.get("citation")
    ]


def to_chat_response(thread_id: str, result: dict[str, Any]) -> ChatResponse:
    """Build the response for one turn.

    Args:
        thread_id: the thread that was run.
        result: whatever `graph.invoke` returned.

    Returns:
        A `ChatResponse`. When the graph interrupted, `status` says which kind of
        decision is needed and `interrupt` carries the payload, so the client
        knows exactly what to ask and where to send the answer.
    """
    payload = interrupt_of(result)
    if payload is not None:
        mode = payload.get("mode", "staff_approve")
        return ChatResponse(
            thread_id=thread_id,
            status=STATUS_FOR_MODE.get(mode, "awaiting_approval"),
            response="",
            interrupt=payload,
            intent=result.get("intent"),
            intent_conf=result.get("intent_conf"),
            trace=list(result.get("trace") or []),
        )

    return ChatResponse(
        thread_id=thread_id,
        status="done",
        response=result.get("response") or "",
        intent=result.get("intent"),
        intent_conf=result.get("intent_conf"),
        escalated=bool(result.get("escalate")),
        ticket_id=result.get("ticket_id"),
        tools_used=list(result.get("tools_used") or []),
        citations=citations_of(result),
        trace=list(result.get("trace") or []),
    )
