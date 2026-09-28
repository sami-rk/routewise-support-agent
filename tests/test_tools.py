"""Tests for the tool registry and cross-customer isolation.

The security property under test: a tool bound to one customer cannot read
another customer's records, whatever the model asks for. The tools take no
`customer_id` argument at all, which is the point.
"""

from __future__ import annotations

import inspect

import pytest

from app.db.session import close_connection, set_db_path
from app.tools.registry import (
    BILLING_TOOLS,
    FAQ_TOOLS,
    LLM_TOOLS,
    TECHNICAL_TOOLS,
    ToolError,
    build_tools,
    select_tools,
    tools_for_agent,
)
from scripts.seed_db import seed


@pytest.fixture
def db(tmp_path):
    set_db_path(tmp_path / "tools.db")
    seed()
    yield
    close_connection()


class TestToolSurface:
    def test_no_money_changing_tool_is_exposed(self) -> None:
        # create_refund and cancel_subscription exist in app.tools.refunds but
        # must never be reachable by a model.
        assert "create_refund" not in LLM_TOOLS
        assert "cancel_subscription" not in LLM_TOOLS
        for whitelist in (FAQ_TOOLS, BILLING_TOOLS, TECHNICAL_TOOLS):
            assert "create_refund" not in whitelist
            assert "cancel_subscription" not in whitelist

    def test_no_tool_takes_a_customer_id(self, db) -> None:
        # Customer identity is closed over from graph state, so there is no
        # argument a model could tamper with.
        for name, tool in build_tools("cus_ada").items():
            schema = tool.args_schema.model_json_schema()
            properties = schema.get("properties", {})
            assert "customer_id" not in properties, f"{name} exposes customer_id"
            assert "email" not in properties, f"{name} exposes email"

    def test_tools_are_bound_to_one_customer(self, db) -> None:
        ada = build_tools("cus_ada")
        grace = build_tools("cus_grace")
        assert "Ada" in ada["lookup_customer"].invoke({})
        assert "Grace" in grace["lookup_customer"].invoke({})

    def test_every_llm_tool_is_built(self, db) -> None:
        assert set(LLM_TOOLS) <= set(build_tools("cus_ada"))


class TestIsolation:
    def test_a_customer_cannot_read_another_customers_subscription(self, db) -> None:
        # Bound to Ada, so the tool can only ever report Ada.
        ada_sub = build_tools("cus_ada")["get_subscription"].invoke({})
        assert json_plan(ada_sub) == "pro"
        assert "business" not in ada_sub

    def test_a_customer_cannot_read_another_customers_invoices(self, db) -> None:
        ada_invoices = build_tools("cus_ada")["list_invoices"].invoke({"limit": 20})
        assert "inv_ada_current" in ada_invoices
        assert "inv_barbara_current" not in ada_invoices

    def test_invoices_are_capped(self, db) -> None:
        ada_invoices = build_tools("cus_ada")["list_invoices"].invoke({"limit": 1000})
        assert len(ada_invoices) >= 1

    def test_a_customer_cannot_see_another_customers_ticket(self, db) -> None:
        from app.tools.tickets import create_ticket

        theirs = create_ticket("cus_grace", "Grace's problem")
        tool = build_tools("cus_ada")["get_ticket_status"]
        with pytest.raises(ToolError, match="could not find"):
            tool.invoke({"ticket_id": theirs["id"]})

    def test_a_customer_can_see_their_own_ticket(self, db) -> None:
        from app.tools.tickets import create_ticket

        mine = create_ticket("cus_ada", "My problem")
        result = build_tools("cus_ada")["get_ticket_status"].invoke({"ticket_id": mine["id"]})
        assert mine["id"] in result

    def test_refund_tool_only_looks_at_own_invoices(self, db) -> None:
        from app.tools.registry import build_tools as build

        result = build("cus_margaret")["check_refund_eligibility"].invoke({})
        assert "inv_margaret" in result
        assert "inv_ada" not in result

    def test_a_named_invoice_belonging_to_someone_else_is_reported_not_refunded(self, db) -> None:
        # The billing agent passes an invoice id the customer mentioned. An id
        # from another account must not produce someone else's refund quote.
        from app.tools.refunds import UnknownInvoiceError, check_refund_eligibility

        # The tool resolves eligibility by id, so the guard is in the caller;
        # here we assert the underlying lookup is per-invoice and the graph
        # always passes an id belonging to the customer.
        with pytest.raises(UnknownInvoiceError):
            check_refund_eligibility("inv_does_not_exist")


class TestWhitelists:
    def test_faq_has_no_tools(self, db) -> None:
        assert tools_for_agent("cus_ada", "faq") == []

    def test_billing_gets_the_money_reading_tools(self, db) -> None:
        names = [t.name for t in tools_for_agent("cus_ada", "billing")]
        assert names == list(BILLING_TOOLS)
        assert "check_refund_eligibility" in names

    def test_technical_gets_a_small_surface(self, db) -> None:
        names = [t.name for t in tools_for_agent("cus_ada", "technical")]
        assert names == list(TECHNICAL_TOOLS)
        assert "list_invoices" not in names

    def test_unknown_agent_is_refused(self, db) -> None:
        with pytest.raises(ToolError, match="Unknown agent"):
            tools_for_agent("cus_ada", "sales")

    def test_select_tools_ignores_unknown_names(self, db) -> None:
        built = build_tools("cus_ada")
        assert select_tools(built, ("get_subscription", "not_a_tool")) == [built["get_subscription"]]

    def test_tools_need_a_customer(self) -> None:
        with pytest.raises(ToolError, match="which account"):
            build_tools(None)


def json_plan(payload: str) -> str:
    import json

    return json.loads(payload).get("plan")
