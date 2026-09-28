"""Customer and subscription lookups.

Every tool here takes the customer id from graph state, never from the model's
output: a tool call is a suggestion by the LLM, and a customer must not be able
to reach another customer's records by naming them in a message. The graph
injects `customer_id`; see `resolve_customer_id`.
"""

from __future__ import annotations

from typing import Any

from ..db.session import query_all, query_one


class UnknownCustomerError(LookupError):
    """No such customer, or the id did not match the one in the request."""


def resolve_customer_id(state_customer_id: str | None, claimed: str | None = None) -> str:
    """Return the id from graph state, ignoring anything the model claimed.

    The model may only narrow a lookup within the caller's own record, so this
    is the single place the id is decided.

    Args:
        state_customer_id: the authenticated customer for this thread.
        claimed: an id or email the model put in the tool arguments.

    Returns:
        The caller's customer id.

    Raises:
        UnknownCustomerError: if the thread has no customer, or the claim points
            at a different one.
    """
    if not state_customer_id:
        raise UnknownCustomerError("No customer is attached to this request")
    if claimed and claimed != state_customer_id:
        raise UnknownCustomerError(
            f"This conversation can only act on behalf of {state_customer_id}"
        )
    return state_customer_id


def get_customer(customer_id: str) -> dict[str, Any] | None:
    """Profile row for a customer, or None."""
    row = query_one("SELECT * FROM customers WHERE id = ?", (customer_id,))
    return dict(row) if row else None


def get_customer_by_email(email: str) -> dict[str, Any] | None:
    """Profile row for an email address, or None."""
    row = query_one("SELECT * FROM customers WHERE email = ? COLLATE NOCASE", (email,))
    return dict(row) if row else None


def get_subscription(customer_id: str) -> dict[str, Any] | None:
    """The customer's active subscription, or None.

    An account has at most one live subscription; a cancelled one is still
    returned so the agent can explain what happened.
    """
    row = query_one(
        "SELECT * FROM subscriptions WHERE customer_id = ? "
        "ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, started_at DESC LIMIT 1",
        (customer_id,),
    )
    return dict(row) if row else None


def get_tickets(customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """A customer's recent tickets, newest first."""
    rows = query_all(
        "SELECT * FROM tickets WHERE customer_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
        (customer_id, max(1, int(limit))),
    )
    return [dict(row) for row in rows]
