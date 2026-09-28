"""Ticket creation and status.

A ticket is a real row with a real id, which is what the original project faked
with `hash()`. The id goes back to the customer and into the handoff message, so
support staff can find it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ..db.session import execute, query_all, query_one

UTC = timezone.utc

# Priority bands, used by the escalation node from the router's urgency.
PRIORITIES = ("low", "normal", "high", "urgent")
CATEGORIES = ("general", "technical", "billing", "account", "escalation")

# Words in a subject that suggest the priority, when the caller does not say.
URGENT_MARKERS = ("data loss", "lost my files", "hacked", "breach", "sue", "lawyer", "fraud", "down")


class UnknownTicketError(LookupError):
    """No such ticket."""


def priority_for(urgency: float | None, subject: str = "") -> str:
    """Map the router's urgency onto a ticket priority.

    Args:
        urgency: 0..1 from the router's score answer, or None.
        subject: the ticket subject, used to bump an otherwise-normal ticket.

    Returns:
        One of `PRIORITIES`.
    """
    lowered = (subject or "").lower()
    if any(marker in lowered for marker in URGENT_MARKERS):
        return "urgent"
    if urgency is None:
        return "normal"
    if urgency >= 0.75:
        return "urgent"
    if urgency >= 0.5:
        return "high"
    if urgency <= 0.25:
        return "low"
    return "normal"


def create_ticket(
    customer_id: str,
    subject: str,
    body: str = "",
    *,
    priority: str | None = None,
    category: str = "general",
    urgency: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create a ticket and return it, including its real id.

    Args:
        customer_id: whose ticket this is. Always from graph state.
        subject: one line describing the problem.
        body: the full description, or the conversation so far.
        priority: a forced priority from `PRIORITIES`, or None to derive it from
            `urgency` and the subject.
        category: one of `CATEGORIES`.
        urgency: 0..1 router urgency, used when priority is not forced.
        now: reference time, for tests.

    Returns:
        The ticket row as a dict, with `ticket_id` as an alias of `id`.
    """
    subject = (subject or "").strip()[:200] or "Support request"
    body = (body or "").strip()
    if priority not in PRIORITIES:
        priority = priority_for(urgency, subject)
    category = category if category in CATEGORIES else "general"

    ticket_id = f"tkt_{uuid.uuid4().hex[:12]}"
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    execute(
        "INSERT INTO tickets (id, customer_id, subject, body, priority, category, status, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)",
        (ticket_id, customer_id, subject, body, priority, category, stamp, stamp),
    )
    ticket = get_ticket(ticket_id)
    assert ticket is not None  # just inserted
    return {**ticket, "ticket_id": ticket_id}


def get_ticket(ticket_id: str) -> dict[str, Any] | None:
    """One ticket by id, or None."""
    row = query_one("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    return dict(row) if row else None


def get_ticket_status(ticket_id: str, *, owner_id: str | None = None) -> dict[str, Any]:
    """Status and last update for a ticket.

    Args:
        ticket_id: the ticket to look up.
        owner_id: when given, the ticket must belong to this customer. A
            mismatch is reported as "not found" rather than "forbidden", so the
            caller cannot use this to discover which ids exist.

    Raises:
        UnknownTicketError: if the ticket does not exist, or does not belong to
            `owner_id`.
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise UnknownTicketError(f"No ticket {ticket_id!r}")
    if owner_id is not None and ticket["customer_id"] != owner_id:
        raise UnknownTicketError(f"No ticket {ticket_id!r}")
    return {
        "ticket_id": ticket["id"],
        "status": ticket["status"],
        "priority": ticket["priority"],
        "category": ticket["category"],
        "subject": ticket["subject"],
        "created_at": ticket["created_at"],
        "updated_at": ticket["updated_at"],
    }


def list_tickets(
    status: str | None = None,
    *,
    limit: int = 50,
    customer_id: str | None = None,
) -> list[dict[str, Any]]:
    """Tickets for the staff queue, or one customer's tickets.

    Args:
        status: filter by status, or None for all.
        limit: how many to return, capped at 200.
        customer_id: restrict to one customer.
    """
    limit = max(1, min(int(limit), 200))
    if customer_id:
        rows = query_all(
            "SELECT * FROM tickets WHERE customer_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (customer_id, limit),
        )
    elif status:
        rows = query_all(
            "SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (status, limit),
        )
    else:
        rows = query_all(
            "SELECT * FROM tickets ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        )
    return [dict(row) for row in rows]


def update_ticket_status(ticket_id: str, status: str, *, now: datetime | None = None) -> dict[str, Any]:
    """Move a ticket along, for the staff queue in the UI.

    Raises:
        UnknownTicketError: if the ticket does not exist.
    """
    if get_ticket(ticket_id) is None:
        raise UnknownTicketError(f"No ticket {ticket_id!r}")
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    execute("UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?", (status, stamp, ticket_id))
    ticket = get_ticket(ticket_id)
    assert ticket is not None
    return ticket
