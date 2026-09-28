"""Invoice lookups.

An invoice is the unit a refund is issued against, so the helpers here are the
ones that decide what a customer can be told about a charge and how old it is.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..db.session import query_all, query_one

UTC = timezone.utc


def parse_timestamp(value: str | datetime | None) -> datetime | None:
    """Parse a SQLite `datetime('now')` string into an aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised timestamp: {value!r}")


def days_since(value: str | datetime | None, *, now: datetime | None = None) -> int | None:
    """Whole days between `value` and now, or None when the date is unusable.

    Truncated, not rounded: a charge at 23:00 yesterday is 0 days old, which is
    inside any window, while rounding would push it to 1 and move the boundary
    a day early.
    """
    when = parse_timestamp(value)
    if when is None:
        return None
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return max(0, (reference - when).days)


def list_invoices(customer_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """A customer's most recent charges, newest first.

    Args:
        customer_id: whose invoices to read. Always from graph state.
        limit: how many to return, capped at 20.
    """
    rows = query_all(
        "SELECT * FROM invoices WHERE customer_id = ? ORDER BY charged_at DESC, id DESC LIMIT ?",
        (customer_id, max(1, min(int(limit), 20))),
    )
    return [dict(row) for row in rows]


def get_invoice(invoice_id: str) -> dict[str, Any] | None:
    """One invoice by id, or None."""
    row = query_one("SELECT * FROM invoices WHERE id = ?", (invoice_id,))
    return dict(row) if row else None


def get_refunds_for_invoice(invoice_id: str) -> list[dict[str, Any]]:
    """Refunds already issued against an invoice."""
    rows = query_all(
        "SELECT * FROM refunds WHERE invoice_id = ? ORDER BY created_at DESC", (invoice_id,)
    )
    return [dict(row) for row in rows]


def refunded_amount(invoice_id: str) -> float:
    """How much of an invoice has already been refunded."""
    row = query_one(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM refunds "
        "WHERE invoice_id = ? AND status = 'succeeded'",
        (invoice_id,),
    )
    return float(row["total"]) if row else 0.0
