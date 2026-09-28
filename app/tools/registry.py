"""The LangChain tools the agents may call, and the per-agent whitelists.

Two rules shape this module:

* `create_refund` and `cancel_subscription` are **not** tools. They change money
  and access, so only `execute_action` runs them, after the approval gate has
  interrupted. The model can propose the action, never perform it.
* Every tool resolves the customer from the bound context rather than from its
  own arguments, so a model that invents a `customer_id` still cannot read
  another customer's invoices.

Each tool is a closure over the customer id, bound per request by
`bind_tools_for(state)`.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

from . import billing, customers, refunds, tickets
from .customers import UnknownCustomerError

# Which tools each agent may call. Kept small: free-tier models handle small
# schemas better, and each tool in the list is a chance to call the wrong thing.
FAQ_TOOLS: tuple[str, ...] = ()
BILLING_TOOLS: tuple[str, ...] = (
    "lookup_customer",
    "get_subscription",
    "list_invoices",
    "check_refund_eligibility",
    "propose_refund",
    "propose_cancellation",
)
TECHNICAL_TOOLS: tuple[str, ...] = ("lookup_customer", "get_subscription")
TICKET_TOOLS: tuple[str, ...] = ("get_ticket_status",)

# Every tool that is safe to expose to an LLM.
LLM_TOOLS: tuple[str, ...] = (
    "lookup_customer",
    "get_subscription",
    "list_invoices",
    "check_refund_eligibility",
    # These record a *proposal* and do nothing themselves. They exist because
    # free models follow a tool schema reliably and an `ACTION: {json}` line in
    # prose not at all.
    "propose_refund",
    "propose_cancellation",
    "get_ticket_status",
)


class ToolError(RuntimeError):
    """A tool could not do what it was asked. Safe to show the customer."""


def _load(customer_id: str) -> Callable[[], str]:
    return lambda: customer_id


def _check_customer_id(customer_id: str | None) -> str:
    if not customer_id:
        raise ToolError("I could not tell which account this is about. Please tell me your email.")
    return customer_id


def build_tools(customer_id: str | None, *, invoice_id: str | None = None) -> dict[str, BaseTool]:
    """Build the read-only tools bound to one customer.

    Args:
        customer_id: the authenticated customer, from graph state.
        invoice_id: the invoice the customer named, if any. `check_refund_eligibility`
            uses it; without it the tool looks for the newest charge inside the
            refund window, which is what a customer usually means.

    Returns:
        Tool name to LangChain tool, ready for `bind_tools`.
    """
    cid = _check_customer_id(customer_id)

    @tool
    def lookup_customer() -> str:
        """Look up the signed-in customer's profile, plan and status.

        Use this when you need the customer's name or account details. There are
        no arguments: the customer is whoever is signed in to this conversation.
        """
        profile = customers.get_customer(cid)
        if profile is None:
            raise ToolError("I could not find an account for this conversation.")
        subscription = customers.get_subscription(cid)
        payload = {
            "name": profile["name"],
            "email": profile["email"],
            "plan": subscription["plan"] if subscription else None,
            "subscription_status": subscription["status"] if subscription else "none",
        }
        return json.dumps(payload)

    @tool
    def get_subscription() -> str:
        """Get the customer's current subscription: plan, price, devices used and allowed, renewal date and status.

        Use this when the customer asks what plan they are on or when their
        subscription renews.
        """
        subscription = customers.get_subscription(cid)
        if subscription is None:
            return json.dumps({"error": "No subscription found on this account."})
        allowed = subscription["devices_allowed"]
        return json.dumps(
            {
                "plan": subscription["plan"],
                "price_monthly": subscription["price_monthly"],
                "devices_used": subscription["devices_used"],
                "devices_allowed": "unlimited" if allowed == 0 else allowed,
                "status": subscription["status"],
                "renews_at": subscription["renews_at"],
            }
        )

    @tool
    def list_invoices(limit: int = 5) -> str:
        """List the customer's recent charges with date, amount and status.

        Args:
            limit: how many invoices to return, 1 to 20.
        """
        rows = billing.list_invoices(cid, limit=limit)
        return json.dumps(
            [
                {
                    "invoice_id": row["id"],
                    "amount": row["amount"],
                    "currency": row["currency"],
                    "status": row["status"],
                    "charged_at": row["charged_at"],
                }
                for row in rows
            ]
        )

    @tool
    def check_refund_eligibility() -> str:
        """Check whether a charge can be refunded under the 14-day policy.

        Call this before promising a refund. It applies the 14-day-from-charge
        rule, says the maximum refundable amount, and says whether a person has
        to approve it. Takes no arguments: it uses the invoice the customer named
        if they named one, otherwise their most recent charge.

        Returns:
            JSON with `eligible`, `reason`, `max_amount` and `needs_approval`.
        """
        target = invoice_id
        if not target:
            found = refunds.find_refundable_invoice(cid)
            if found is None:
                # Nothing recent is eligible; report the newest charge so the
                # agent can explain the window rather than shrug.
                recent = billing.list_invoices(cid, limit=1)
                if not recent:
                    return json.dumps({"error": "No invoices found on this account."})
                try:
                    return json.dumps(refunds.check_refund_eligibility(recent[0]["id"]))
                except refunds.UnknownInvoiceError:
                    return json.dumps({"error": "No invoices found on this account."})
            return json.dumps(found)
        return json.dumps(refunds.check_refund_eligibility(target))

    @tool
    def get_ticket_status(ticket_id: str) -> str:
        """Get the status and last update of one of this customer's support tickets.

        Args:
            ticket_id: the ticket id, which looks like `tkt_abc123`. Only this
                customer's own tickets can be read.
        """
        try:
            return json.dumps(tickets.get_ticket_status(ticket_id, owner_id=cid))
        except tickets.UnknownTicketError:
            raise ToolError(f"I could not find a ticket with the id {ticket_id!r}.")

    @tool
    def propose_refund(invoice_id: str, amount: float, reason: str) -> str:
        """Propose a refund of an invoice. This does NOT issue it; the system
        decides, and a person approves anything above the limit.

        Call this only when the customer has actually asked for a refund, you have
        already checked eligibility with check_refund_eligibility, and it was
        eligible. Never call it to promise a refund you have not checked.

        Args:
            invoice_id: the invoice to refund.
            amount: how much to refund, no more than the eligible amount.
            reason: why, in the customer's own terms.
        """
        return json.dumps({"status": "proposed", "type": "refund", "invoice_id": invoice_id})

    @tool
    def propose_cancellation() -> str:
        """Propose cancelling this customer's subscription.

        Call this only when the customer has asked to cancel. This does NOT cancel
        anything: the customer is asked to confirm first, and the system does the
        rest.
        """
        return json.dumps({"status": "proposed", "type": "cancel"})

    return {
        "lookup_customer": lookup_customer,
        "get_subscription": get_subscription,
        "list_invoices": list_invoices,
        "check_refund_eligibility": check_refund_eligibility,
        "propose_refund": propose_refund,
        "propose_cancellation": propose_cancellation,
        "get_ticket_status": get_ticket_status,
    }


def select_tools(all_tools: dict[str, BaseTool], allowed: tuple[str, ...]) -> list[BaseTool]:
    """The subset of tools an agent may call, in the order given."""
    return [all_tools[name] for name in allowed if name in all_tools]


def tools_for_agent(
    customer_id: str | None,
    agent: str,
    *,
    invoice_id: str | None = None,
) -> list[BaseTool]:
    """The tools one agent is allowed, ready for `bind_tools`.

    Args:
        customer_id: the authenticated customer.
        agent: "faq", "billing", "technical" or "tickets".
        invoice_id: the invoice the customer named, if any.

    Raises:
        ToolError: for an unknown agent name.
    """
    whitelists = {
        "faq": FAQ_TOOLS,
        "billing": BILLING_TOOLS,
        "technical": TECHNICAL_TOOLS,
        "tickets": TICKET_TOOLS,
    }
    if agent not in whitelists:
        raise ToolError(f"Unknown agent {agent!r}")
    built = build_tools(customer_id, invoice_id=invoice_id)
    return select_tools(built, whitelists[agent])


__all__ = [
    "BILLING_TOOLS",
    "FAQ_TOOLS",
    "LLM_TOOLS",
    "TECHNICAL_TOOLS",
    "TICKET_TOOLS",
    "ToolError",
    "UnknownCustomerError",
    "build_tools",
    "select_tools",
    "tools_for_agent",
]
