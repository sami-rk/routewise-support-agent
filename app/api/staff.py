"""Staff endpoints: refund approvals, the ticket queue, metrics and traces.

All of these sit behind `require_admin`, because they expose other customers'
records or let a caller move money.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langgraph.types import Command

from ..db.session import query_all
from ..observability.metrics import collect
from ..observability.router_log import recent_decisions
from ..observability.tracing import get_thread_trace
from ..tools.tickets import UnknownTicketError, list_tickets, update_ticket_status
from .deps import config_for, graph_dep, require_admin
from .chat import interrupt_of_snapshot
from .models import ApprovalDecision
from .responses import to_chat_response

router = APIRouter(
    tags=["staff"],
    dependencies=[Depends(require_admin)],
)

# Where a pause is recorded while it waits for a decision. LangGraph's
# checkpointer holds the thread, but the queue needs a queryable row.
PENDING_STATUS = "awaiting"


def _pending_row(thread_id: str, customer_id: str, payload: dict[str, Any]) -> None:
    from ..db.session import execute

    execute(
        "INSERT INTO pending_actions (thread_id, customer_id, action, payload, mode, status) "
        "VALUES (?, ?, ?, ?, ?, 'awaiting') "
        "ON CONFLICT(thread_id) DO UPDATE SET payload = excluded.payload, "
        "mode = excluded.mode, status = 'awaiting', resolved_at = NULL",
        (
            thread_id,
            customer_id,
            str(payload.get("type", "unknown")),
            __import__("json").dumps(payload, default=str),
            str(payload.get("mode", "staff_approve")),
        ),
    )


def _resolve_pending(thread_id: str, decision: str, note: str | None) -> None:
    from ..db.session import execute

    execute(
        "UPDATE pending_actions SET status = ?, resolved_at = datetime('now') WHERE thread_id = ?",
        (decision, thread_id),
    )


@router.get("/approvals/pending")
def pending_approvals() -> list[dict[str, Any]]:
    """Threads waiting for a staff decision, oldest first."""
    return [dict(row) for row in query_all(
        "SELECT * FROM pending_actions WHERE status = 'awaiting' ORDER BY created_at ASC"
    )]


@router.post("/approvals/{thread_id}")
def decide(thread_id: str, body: ApprovalDecision, graph=Depends(graph_dep)) -> dict[str, Any]:
    """Approve or reject a paused refund.

    The decision is applied to the thread with `Command(resume=...)`, so the graph
    continues from where it stopped. `approved_by` is recorded on the refund.
    """
    snapshot = graph.get_state(config_for(thread_id))
    interrupt = interrupt_of_snapshot(snapshot)
    if interrupt is None or interrupt.get("mode") != "staff_approve":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This thread is not waiting for a staff approval.",
        )

    result = graph.invoke(
        Command(
            resume={
                "decision": body.decision,
                "note": body.note,
                "approved_by": body.approved_by or "staff:api",
            }
        ),
        config_for(thread_id),
    )
    _resolve_pending(thread_id, body.decision, body.note)
    return to_chat_response(thread_id, result).model_dump()


@router.get("/tickets")
def tickets(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """The ticket queue, newest first."""
    return list_tickets(status=status_filter, limit=limit)


@router.get("/tickets/{ticket_id}")
def ticket_detail(ticket_id: str) -> dict[str, Any]:
    """One ticket."""
    from ..tools.tickets import get_ticket

    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No ticket {ticket_id!r}."
        )
    return ticket


@router.patch("/tickets/{ticket_id}")
def set_ticket_status(ticket_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Move a ticket along the queue."""
    new_status = body.get("status")
    if not new_status:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A status is required."
        )
    try:
        return update_ticket_status(ticket_id, new_status)
    except UnknownTicketError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.get("/traces/{thread_id}")
def traces(thread_id: str) -> dict[str, Any]:
    """The per-node trace and the router's decisions for one thread."""
    return {
        "thread_id": thread_id,
        "trace": get_thread_trace(thread_id),
        "router_decisions": recent_decisions(thread_id),
    }


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    """Aggregated metrics, all read from the database."""
    return collect()
