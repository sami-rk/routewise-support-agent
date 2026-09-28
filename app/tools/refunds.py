"""Refund policy: eligibility, issuing refunds, and cancelling subscriptions.

`create_refund` and `cancel_subscription` are deliberately not exposed to the
LLM. They change money and access, so only `execute_action` runs them, after the
approval gate has interrupted for a decision. Everything the model is allowed to
call here is read-only.

The policy itself: a charge is refundable within `REFUND_WINDOW_DAYS` (14) of the
charge date, and a refund of `REFUND_AUTO_LIMIT` (20) or less is issued
immediately. Larger refunds need a person to approve.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ..config import Settings, get_settings
from ..db.session import execute, query_one
from .billing import days_since, get_invoice, get_refunds_for_invoice, refunded_amount

UTC = timezone.utc


class RefundError(RuntimeError):
    """A refund was refused. The message is safe to show the customer."""


class UnknownInvoiceError(RefundError):
    """No such invoice."""


def check_refund_eligibility(
    invoice_id: str,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Decide whether an invoice can be refunded, and for how much.

    Applies the 14-day-from-charge rule and subtracts anything already refunded.

    Args:
        invoice_id: the invoice to check.
        settings: policy values, defaulting to the process settings.
        now: reference time, for tests.

    Returns:
        A dict with `eligible`, `reason`, `max_amount`, plus the invoice detail
        (`amount`, `age_days`, `window_days`, `already_refunded`, `needs_approval`)
        so the agent can explain the answer and cite the policy.

    Raises:
        UnknownInvoiceError: if the invoice does not exist.
    """
    settings = settings or get_settings()
    invoice = get_invoice(invoice_id)
    if invoice is None:
        raise UnknownInvoiceError(f"No invoice {invoice_id!r}")

    amount = float(invoice["amount"])
    age = days_since(invoice["charged_at"], now=now)
    window = settings.refund_window_days
    already = refunded_amount(invoice_id)
    remaining = round(amount - already, 2)

    detail: dict[str, Any] = {
        "invoice_id": invoice_id,
        "amount": amount,
        "currency": invoice.get("currency", "USD"),
        "age_days": age,
        "window_days": window,
        "already_refunded": already,
        "auto_limit": settings.refund_auto_limit,
        "charged_at": invoice["charged_at"],
    }

    if invoice.get("status") not in ("paid", None):
        return {**detail, "eligible": False, "max_amount": 0.0, "needs_approval": False,
                "reason": f"That invoice is {invoice['status']}, not a settled charge."}

    if age is None:
        return {**detail, "eligible": False, "max_amount": 0.0, "needs_approval": False,
                "reason": "I could not read the charge date, so I cannot check the refund window."}

    if already >= amount:
        return {**detail, "eligible": False, "max_amount": 0.0, "needs_approval": False,
                "reason": "This charge has already been refunded in full."}

    if age > window:
        return {
            **detail,
            "eligible": False,
            "max_amount": 0.0,
            "needs_approval": False,
            "reason": (
                f"This charge is {age} days old. Refunds are available within {window} days "
                f"of the charge date, so this one is outside the window."
            ),
        }

    if remaining <= 0:
        return {**detail, "eligible": False, "max_amount": 0.0, "needs_approval": False,
                "reason": "This charge has already been refunded in full."}

    needs_approval = remaining > settings.refund_auto_limit
    reason = (
        f"This charge is {age} days old, inside the {window}-day refund window, so it is "
        f"eligible for up to ${remaining:.2f}."
    )
    if needs_approval:
        reason += (
            f" Because it is above ${settings.refund_auto_limit:.0f}, a support specialist "
            "has to approve it first."
        )
    return {**detail, "eligible": True, "max_amount": remaining, "needs_approval": needs_approval,
            "reason": reason}


def find_refundable_invoice(
    customer_id: str,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """The customer's newest charge that is inside the refund window.

    The agent calls this when a customer says "I want a refund" without naming an
    invoice, which is the common case.

    Args:
        customer_id: whose invoices to check. Always from graph state.
        settings: policy values.
        now: reference time, for tests.

    Returns:
        The eligibility result for the first eligible charge, or None.
    """
    from .billing import list_invoices

    settings = settings or get_settings()
    for invoice in list_invoices(customer_id, limit=10):
        try:
            result = check_refund_eligibility(invoice["id"], settings=settings, now=now)
        except UnknownInvoiceError:
            continue
        if result["eligible"]:
            return result
    return None


def create_refund(
    invoice_id: str,
    amount: float,
    reason: str,
    *,
    approved_by: str | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Refund part or all of an invoice.

    Not exposed to the LLM: `execute_action` calls this after the approval gate.

    Args:
        invoice_id: the invoice to refund.
        amount: how much to refund. Must be positive and no more than the
            remaining refundable amount.
        reason: why, for the audit trail.
        approved_by: who approved it, or None for an automatic refund.
        settings: policy values.
        now: reference time, for tests.

    Returns:
        The refund row as a dict.

    Raises:
        UnknownInvoiceError: if the invoice does not exist.
        RefundError: if the invoice is not eligible, or the amount is invalid.
    """
    settings = settings or get_settings()
    eligibility = check_refund_eligibility(invoice_id, settings=settings, now=now)

    if not eligibility["eligible"]:
        raise RefundError(eligibility["reason"])

    amount = round(float(amount), 2)
    if amount <= 0:
        raise RefundError("A refund has to be for a positive amount.")
    if amount > eligibility["max_amount"] + 0.001:
        raise RefundError(
            f"${amount:.2f} is more than the ${eligibility['max_amount']:.2f} that can still "
            "be refunded on that charge."
        )
    if not approved_by and amount > settings.refund_auto_limit:
        raise RefundError(
            f"${amount:.2f} is above the ${settings.refund_auto_limit:.0f} automatic limit, "
            "so it needs a support specialist to approve it."
        )

    refund_id = f"rf_{uuid.uuid4().hex[:12]}"
    execute(
        "INSERT INTO refunds (id, invoice_id, amount, reason, status, approved_by, created_at) "
        "VALUES (?, ?, ?, ?, 'succeeded', ?, ?)",
        (refund_id, invoice_id, amount, reason, approved_by, (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")),
    )
    return {
        "refund_id": refund_id,
        "invoice_id": invoice_id,
        "amount": amount,
        "reason": reason,
        "status": "succeeded",
        "approved_by": approved_by or "automatic",
    }


def cancel_subscription(
    customer_id: str,
    *,
    approved_by: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Cancel a subscription at the end of the paid period.

    Not exposed to the LLM: `execute_action` calls this after the customer
    confirms.

    Args:
        customer_id: whose subscription to cancel.
        approved_by: who asked, for the audit trail.
        now: reference time, for tests.

    Returns:
        A summary dict.

    Raises:
        RefundError: if there is no subscription, or it is already cancelled.
    """
    from .customers import get_subscription

    subscription = get_subscription(customer_id)
    if subscription is None:
        raise RefundError("I could not find a subscription on this account.")
    if subscription["status"] == "cancelled":
        raise RefundError("This subscription is already cancelled.")

    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    execute(
        "UPDATE subscriptions SET status = 'cancelled', renews_at = NULL WHERE id = ?",
        (subscription["id"],),
    )
    return {
        "subscription_id": subscription["id"],
        "plan": subscription["plan"],
        "status": "cancelled",
        "renews_at": subscription["renews_at"],
        "cancelled_by": approved_by or "customer",
        "cancelled_at": stamp,
    }
