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

import json
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

    if mode == "auto":
        # Inside the window and under the limit: no interruption, no waiting on
        # a person. This is the whole point of REFUND_AUTO_LIMIT.
        return {
            "approval": "approved",
            "approval_note": "within the automatic limit",
            "mode": "auto",
            "approved_by": "automatic",
            "interrupt": None,
        }

    record_pending(state, payload)
    resume_value = interrupt(payload)
    decision, note, by = read_resume(resume_value)

    resolve_pending(state.get("thread_id") or "", decision, note)

    return {
        "approval": decision,
        "approval_note": note,
        "approved_by": by or default_approver(mode),
        "interrupt": None,
        "mode": mode,
    }


def record_pending(state: SupportState, payload: dict[str, Any]) -> None:
    """Note that a thread is waiting, so the staff queue can list it.

    Written here, in the gate, rather than in an API handler, so the CLI and the
    SSE stream produce the same queue as `POST /chat`.
    """
    execute(
        "INSERT INTO pending_actions (thread_id, customer_id, action, payload, mode, status) "
        "VALUES (?, ?, ?, ?, ?, 'awaiting') "
        "ON CONFLICT(thread_id) DO UPDATE SET payload = excluded.payload, "
        "mode = excluded.mode, status = 'awaiting', resolved_at = NULL",
        (
            state.get("thread_id") or "unknown",
            state.get("customer_id") or "",
            str(payload.get("type", "unknown")),
            json.dumps(payload, default=str),
            str(payload.get("mode", "staff_approve")),
        ),
    )


def resolve_pending(thread_id: str, decision: str, note: str | None = None) -> None:
    """Mark a queued decision as made."""
    if not thread_id:
        return
    execute(
        "UPDATE pending_actions SET status = ?, resolved_at = datetime('now') WHERE thread_id = ?",
        (decision, thread_id),
    )


def default_approver(mode: str) -> str:
    """Who to record as approving, when the caller did not say.

    A refund approved through the staff endpoint is a person, not the system, so
    the audit trail says so rather than "automatic".
    """
    return "staff:api" if mode == "staff_approve" else "customer"


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


def read_resume(value: Any) -> tuple[str, str | None, str | None]:
    """Read the decision, note and approver out of whatever was resumed with.

    Accepts `{"decision": "approved"}`, a bare string, or the client's
    `{"approved": true}` shape, because the API, the CLI and the UI send
    different things. Anything unrecognised is a rejection, which is the safe
    default: a malformed resume must not run a refund.

    Returns:
        `(decision, note, approved_by)`.
    """
    if value is None:
        return "rejected", "no decision was given", None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("approved", "approve", "yes", "true", "confirm"):
            return "approved", None, None
        return "rejected", value, None
    if isinstance(value, dict):
        decision = value.get("decision") or value.get("status")
        note = value.get("note")
        who = value.get("approved_by") or value.get("by")
        if decision is not None:
            resolved, _, _ = read_resume(str(decision))
            return resolved, note, who
        if "approved" in value:
            return (("approved", note, who) if value["approved"] else ("rejected", note, who))
        if "confirmed" in value:
            return (("approved", note, who) if value["confirmed"] else ("rejected", note, who))
    return "rejected", f"could not understand the decision: {value!r}", None


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
    """Hand the conversation to a person.

    A ticket is created by the `create_ticket` node that follows, so the id is not
    known here. The reply is therefore written without it and `respond` appends
    the reference once the ticket exists.

    The model's prose is used only when it looks like prose. Observed against the
    free models: a reply of "User Safety: safe" came back instead of a handoff,
    and a ticket id was echoed back as the literal placeholder. Neither may reach
    a customer, so both fall back to a fixed message.
    """
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
        text = usable_handoff(message_text(response))
    except LLMUnavailableError:
        # Never fail a handoff because the prose model is busy.
        text = ""

    return {
        "response": text or HANDOFF_FALLBACK,
        "escalate": True,
        "escalated_reason": state.get("needs_human_reason") or "router_decision",
    }


# Used when the model does not produce usable prose. Short, warm, and true.
HANDOFF_FALLBACK = (
    "I am sorry you are dealing with this. I have passed your request to our support "
    "team, and a person will follow up in this conversation."
)

# Shortest text accepted as a handoff. The free models occasionally answer with a
# fragment or a status token rather than a sentence.
MIN_HANDOFF_CHARS = 40

# Phrases that mean the model emitted something other than a handoff.
HANDOFF_REJECT = (
    "ticket_id",
    "user safety",
    "safe\n",
    "i'm sorry, but i cannot",
    "i cannot assist",
    "as an ai",
)


def usable_handoff(text: str) -> str:
    """Whether a model's handoff can be shown to a customer as-is."""
    candidate = (text or "").strip()
    if len(candidate) < MIN_HANDOFF_CHARS:
        return ""
    lowered = candidate.lower()
    if any(marker in lowered for marker in HANDOFF_REJECT):
        return ""
    return candidate


def respond(state: SupportState) -> dict[str, Any]:
    """Finalise the reply: make sure it exists, and add what it should cite.

    Runs after `create_ticket`, so an escalated turn knows its ticket id here and
    can put the reference in the reply. The handoff written by `escalate` was
    composed before the ticket existed.
    """
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

    ticket_id = state.get("ticket_id")
    if state.get("escalate") and ticket_id and ticket_id not in response:
        response = f"{response}\n\nYour reference is {ticket_id}."

    return {"response": response}
