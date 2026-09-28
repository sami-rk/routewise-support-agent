"""The approval gate, the action runner, tickets, escalation and the reply.

This is where the graph touches money. The order matters and is the point of the
design:

* `approval_gate` calls `interrupt()` and returns. Nothing runs yet.
* The API surfaces the interrupt to a person (or back to the customer), and the
  thread is resumed with `Command(resume={"decision": ...})`.
* Only `execute_action` then calls `create_refund` or `cancel_subscription`, and
  only after re-checking the eligibility itself rather than trusting the number
  the model proposed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...config import Settings, get_settings
from ...db.session import execute, query_one
from ...models.llm import LLMUnavailableError
from ...models.llm_runner import invoke_with_fallback
from ...tools import refunds, tickets
from ...tools.refunds import RefundError
from .. import prompts
from ..state import SupportState
from .shared import friendly_error, message_text

logger = logging.getLogger(__name__)

# What a `staff_approve` interrupt looks like to the reviewer.
REFUND_REASON = "Refund requested"
CANCEL_REASON = "Cancellation requested"


def approval_gate(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Pause the turn until a person, or the customer, decides.

    LangGraph's `interrupt()` suspends the graph here: the run returns an
    `__interrupt__` payload to the caller, which is shown in the UI. The same
    thread is then resumed with `Command(resume=...)`, and this node runs again
    from the top, with the decision available the second time.

    Which mode applies is decided by the action, not by the node:

    * `customer_confirm` for a cancellation, because the customer has to agree
      before their access changes;
    * `staff_approve` for a refund, but only when it is above `REFUND_AUTO_LIMIT`
      or otherwise needs a person. A small refund inside the window is automatic
      and never interrupts.
    """
    from langgraph.types import interrupt

    action = state.get("pending_action")
    if not action:
        return {"approval": "approved", "approval_note": "nothing to approve"}

    decision = state.get("approval")
    if decision is not None:
        # Second pass: the thread was resumed with a decision already recorded.
        return {"approval_note": state.get("approval_note")}

    mode, payload = build_interrupt_payload(action, state, settings)

    resume_value = interrupt(payload)
    decision, note = read_resume(resume_value)

    return {
        "approval": decision,
        "approval_note": note,
        "interrupt": None,
        "mode": mode,
    }


def build_interrupt_payload(
    action: dict[str, Any],
    state: SupportState,
    settings: Settings | None = None,
) -> tuple[str, dict[str, Any]]:
    """Decide the interrupt mode and build what the reviewer is shown."""
    settings = settings or get_settings()
    kind = action.get("type")

    if kind == "cancel":
        return "customer_confirm", {
            "mode": "customer_confirm",
            "type": "cancel",
            "thread_id": state.get("thread_id"),
            "customer_id": state.get("customer_id"),
            "question": "Shall I cancel your subscription? You will keep access until the end of the period you have paid for.",
        }

    amount = float(action.get("amount") or 0.0)
    invoice_id = action.get("invoice_id")
    eligibility = None
    try:
        eligibility = refunds.check_refund_eligibility(str(invoice_id), settings=settings)
    except RefundError as exc:
        logger.info("refund interrupt for an ineligible invoice: %s", exc)

    if eligibility and not eligibility["needs_approval"]:
        # Inside the window and small enough to issue without a person.
        return "auto", {"mode": "auto", "type": "refund", "invoice_id": invoice_id, "amount": amount}

    return "staff_approve", {
        "mode": "staff_approve",
        "type": "refund",
        "thread_id": state.get("thread_id"),
        "customer_id": state.get("customer_id"),
        "invoice_id": invoice_id,
        "amount": amount,
        "reason": action.get("reason") or REFUND_REASON,
        "eligibility": eligibility,
        "auto_limit": settings.refund_auto_limit,
    }


def read_resume(value: Any) -> tuple[str, str | None]:
    """Read the decision out of whatever the caller resumed with.

    Accepts `{"decision": "approved"}`, a bare string, or the client's
    `{"approved": true}` shape, because the API and the CLI send different
    things. Anything unrecognised is a rejection, which is the safe default: a
    malformed resume must not run a refund.
    """
    if value is None:
        return "rejected", "no decision was given"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("approved", "approve", "yes", "true", "confirm"):
            return "approved", None
        return "rejected", value
    if isinstance(value, dict):
        decision = value.get("decision") or value.get("status")
        note = value.get("note")
        if decision is not None:
            resolved, _ = read_resume(str(decision))
            return resolved, note
        if "approved" in value:
            return ("approved", note) if value["approved"] else ("rejected", note)
        if "confirmed" in value:
            return ("approved", note) if value["confirmed"] else ("rejected", note)
    return "rejected", f"could not understand the decision: {value!r}"


def execute_action(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Run the approved action, re-checking the rules first.

    The model proposed an amount; this node decides what is actually refunded.
    A refund is capped at what `check_refund_eligibility` says is still
    refundable, so a model cannot refund more than the policy allows by asking
    for more.
    """
    settings = settings or get_settings()
    action = state.get("pending_action")
    if not action:
        return {"action_result": None}
    if state.get("approval") != "approved":
        return {"action_result": {"ok": False, "reason": "not approved"}}

    customer_id = state.get("customer_id")
    kind = action.get("type")

    try:
        if kind == "refund":
            invoice_id = str(action.get("invoice_id") or "")
            eligibility = refunds.check_refund_eligibility(invoice_id, settings=settings)
            if not eligibility["eligible"]:
                return {
                    "action_result": {"ok": False, "reason": eligibility["reason"]},
                    "response": eligibility["reason"],
                }
            amount = min(float(action.get("amount") or 0.0), eligibility["max_amount"])
            result = refunds.create_refund(
                invoice_id,
                amount,
                action.get("reason") or REFUND_REASON,
                approved_by=state.get("approved_by"),
                settings=settings,
            )
            return {
                "action_result": {"ok": True, **result},
                "response": (
                    f"I have refunded ${result['amount']:.2f} to your original payment method. "
                    "Your bank usually shows it within 5 to 10 business days "
                    "[cancellation_refunds.md > Refunds]."
                ),
            }

        if kind == "cancel":
            result = refunds.cancel_subscription(
                customer_id or "", approved_by=state.get("approved_by")
            )
            return {
                "action_result": {"ok": True, **result},
                "response": (
                    "Your subscription is cancelled. You keep access until the end of the "
                    "period you have already paid for, and your files stay on your devices "
                    "[cancellation_refunds.md > Cancelling your subscription]."
                ),
            }
    except RefundError as exc:
        return {"action_result": {"ok": False, "reason": str(exc)}, "response": str(exc)}

    return {"action_result": {"ok": False, "reason": f"unknown action {kind!r}"}}


def create_ticket_node(state: SupportState) -> dict[str, Any]:
    """Create a real ticket and put its id in the state."""
    action = state.get("pending_action") or {}
    subject = action.get("subject") or default_subject(state)
    ticket = tickets.create_ticket(
        state.get("customer_id") or "",
        subject,
        body=action.get("body") or state.get("user_input") or "",
        category="escalation" if state.get("escalate") else "general",
        urgency=state.get("urgency"),
    )
    return {"ticket_id": ticket["id"], "ticket": ticket, "wants_ticket": False}


def default_subject(state: SupportState) -> str:
    """A subject derived from the turn, when the model did not supply one."""
    message = (state.get("user_input") or "").strip()
    subject = message.splitlines()[0] if message else "Support request"
    return subject[:120] or "Support request"


def escalate(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Hand the conversation to a person, and open a ticket for them.

    Replaces the original project's canned message and its `hash()` case id with
    a real ticket and a reply written for this customer.
    """
    ticket = state.get("ticket")
    ticket_id = state.get("ticket_id") or (ticket or {}).get("id") or ""

    # A person is waiting, so this reply has to be right even if the LLM is not.
    fallback = (
        f"I have passed this to our support team, who will follow up in this "
        f"conversation. Your reference is {ticket_id}."
    )
    try:
        response = invoke_with_fallback(
            [
                SystemMessage(prompts.escalation_prompt(state.get("customer_facts") or {}, state)),
                HumanMessage(state.get("user_input") or ""),
            ],
            role="escalate",
            settings=settings,
            run_id=state.get("run_id"),
            thread_id=state.get("thread_id"),
        )
        text = message_text(response) or fallback
    except LLMUnavailableError:
        # Never fail a handoff because the prose model is busy.
        text = fallback

    return {
        "response": text,
        "escalate": True,
        "escalated_reason": state.get("needs_human_reason") or "router_decision",
        "ticket_id": ticket_id or None,
    }


def respond(state: SupportState) -> dict[str, Any]:
    """Finalise the reply: make sure it exists, and add the sources used."""
    from .shared import cited

    response = (state.get("response") or "").strip()
    if not response:
        response = (
            "I do not have a confident answer for that one. I can open a ticket so "
            "someone can look at it properly, if you would like."
        )

    if not state.get("unsafe"):
        sources = [p["citation"] for p in (state.get("retrieved") or []) if p.get("citation")]
        response = cited(response, sources)

    return {"response": response}
