"""The approval gate: refunds and cancellations pausing for a decision.

`interrupt()` is the whole point of this milestone, so the tests drive the graph
the way a real client does — `invoke`, read the interrupt out, then resume the
same thread with `Command(resume=...)` — and check that nothing moved money
before the decision arrived.
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from app.agent.graph import build_graph, thread_config
from app.db.session import close_connection, query_all, query_one
from app.agent.nodes.actions import read_resume
from tests.conftest import user_turn
from tests.stub_llm import ScriptedLLM

REFUND_REPLY = (
    "That charge is 5 days old, so it is inside the 14-day window. "
    'ACTION: {"type": "refund", "invoice_id": "inv_ada_current", "amount": 19.0}'
)
BIG_REFUND_REPLY = (
    'ACTION: {"type": "refund", "invoice_id": "inv_barbara_current", "amount": 49.0}'
)
CANCEL_REPLY = 'Sure, I can cancel that. ACTION: {"type": "cancel"}'


@pytest.fixture
def persistent_graph(seeded, knowledge_base):
    """A graph with a SQLite checkpointer, so interrupts can be resumed."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(seeded.checkpoint_db_path)) as saver:
        yield build_graph(checkpointer=saver, settings=seeded)


def interrupt_of(result: dict) -> dict | None:
    """The interrupt payload, whichever key LangGraph put it under."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


class TestSmallRefundIsAutomatic:
    def test_a_refund_under_the_limit_runs_without_asking_anyone(
        self, persistent_graph
    ) -> None:
        with ScriptedLLM([REFUND_REPLY, "I have refunded $19.00 to your card."]):
            result = persistent_graph.invoke(
                user_turn("I was charged 5 days ago and want a refund"),
                thread_config("t-auto"),
            )
        assert result.get("__interrupt__") is None, "a small refund must not interrupt"
        assert result["action_result"]["ok"] is True
        assert result["action_result"]["amount"] == 19.0

    def test_the_refund_is_really_recorded(self, persistent_graph) -> None:
        with ScriptedLLM([REFUND_REPLY, "Refunded."]):
            persistent_graph.invoke(
                user_turn("I was charged 5 days ago and want a refund"), thread_config("t-auto2")
            )
        rows = query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_ada_current'")
        assert len(rows) == 1
        assert rows[0]["amount"] == 19.0
        assert rows[0]["approved_by"] == "automatic"


class TestLargeRefundNeedsStaffApproval:
    def test_a_large_refund_pauses_at_the_interrupt(self, persistent_graph) -> None:
        with ScriptedLLM([BIG_REFUND_REPLY, "It is with our team now."]):
            result = persistent_graph.invoke(
                user_turn("I want a refund of the 49 dollars", customer_id="cus_barbara"),
                thread_config("t-staff"),
            )
        payload = interrupt_of(result)
        assert payload is not None, "a refund above the limit must pause"
        assert payload["mode"] == "staff_approve"
        assert payload["type"] == "refund"
        assert payload["amount"] == 49.0
        assert payload["invoice_id"] == "inv_barbara_current"

    def test_nothing_is_refunded_before_the_decision(self, persistent_graph) -> None:
        with ScriptedLLM([BIG_REFUND_REPLY, "It is with our team now."]):
            persistent_graph.invoke(
                user_turn("I want a refund of the 49 dollars", customer_id="cus_barbara"),
                thread_config("t-staff2"),
            )
        assert query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_barbara_current'") == []

    def test_approving_runs_the_refund(self, persistent_graph) -> None:
        with ScriptedLLM([BIG_REFUND_REPLY, "It is with our team now.", "Your refund is on its way."]):
            persistent_graph.invoke(
                user_turn("I want a refund of the 49 dollars", customer_id="cus_barbara"),
                thread_config("t-approve"),
            )
            result = persistent_graph.invoke(
                Command(resume={"decision": "approved", "note": "ok"}),
                thread_config("t-approve"),
            )
        assert result["action_result"]["ok"] is True
        rows = query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_barbara_current'")
        assert len(rows) == 1
        assert rows[0]["approved_by"] == "staff:api"

    def test_rejecting_refunds_nothing(self, persistent_graph) -> None:
        with ScriptedLLM([BIG_REFUND_REPLY, "It is with our team now.", "Understood."]):
            persistent_graph.invoke(
                user_turn("I want a refund of the 49 dollars", customer_id="cus_barbara"),
                thread_config("t-reject"),
            )
            result = persistent_graph.invoke(
                Command(resume={"decision": "rejected", "note": "outside policy"}),
                thread_config("t-reject"),
            )
        assert result.get("action_result") in (None, {"ok": False})
        assert query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_barbara_current'") == []


class TestCancellationNeedsTheCustomer:
    def test_cancelling_asks_for_confirmation(self, persistent_graph) -> None:
        with ScriptedLLM([CANCEL_REPLY, "Cancelling now."]):
            result = persistent_graph.invoke(
                user_turn("Cancel my subscription"), thread_config("t-cancel")
            )
        payload = interrupt_of(result)
        assert payload is not None
        assert payload["mode"] == "customer_confirm"
        assert "cancel" in payload["question"].lower()

    def test_the_subscription_is_untouched_until_confirmed(self, persistent_graph) -> None:
        with ScriptedLLM([CANCEL_REPLY, "Cancelling now."]):
            persistent_graph.invoke(user_turn("Cancel my subscription"), thread_config("t-cancel2"))
        row = query_one("SELECT status FROM subscriptions WHERE id = 'sub_ada'")
        assert row["status"] == "active"

    def test_confirming_cancels(self, persistent_graph) -> None:
        with ScriptedLLM([CANCEL_REPLY, "Cancelling now.", "Your subscription is cancelled."]):
            persistent_graph.invoke(user_turn("Cancel my subscription"), thread_config("t-confirm"))
            result = persistent_graph.invoke(
                Command(resume={"decision": "approved"}), thread_config("t-confirm")
            )
        assert result["action_result"]["ok"] is True
        row = query_one("SELECT status FROM subscriptions WHERE id = 'sub_ada'")
        assert row["status"] == "cancelled"

    def test_declining_leaves_the_subscription_alone(self, persistent_graph) -> None:
        with ScriptedLLM([CANCEL_REPLY, "Cancelling now.", "No problem, I left it as it was."]):
            persistent_graph.invoke(user_turn("Cancel my subscription"), thread_config("t-decline"))
            persistent_graph.invoke(
                Command(resume={"decision": "rejected"}), thread_config("t-decline")
            )
        row = query_one("SELECT status FROM subscriptions WHERE id = 'sub_ada'")
        assert row["status"] == "active"


class TestResumeParsing:
    @pytest.mark.parametrize(
        "value",
        ["approved", "approve", "yes", "true", {"decision": "approved"}, {"approved": True}],
    )
    def test_approvals_are_recognised(self, value) -> None:
        assert read_resume(value)[0] == "approved"

    @pytest.mark.parametrize(
        "value", ["rejected", "no", {"decision": "rejected"}, {"approved": False}]
    )
    def test_rejections_are_recognised(self, value) -> None:
        assert read_resume(value)[0] == "rejected"

    @pytest.mark.parametrize("value", [None, {}, {"nonsense": 1}, 42])
    def test_anything_unrecognised_is_a_rejection(self, value) -> None:
        # A malformed resume must never run a refund.
        assert read_resume(value)[0] == "rejected"

    def test_a_note_is_kept(self) -> None:
        assert read_resume({"decision": "approved", "note": "manager ok"})[1] == "manager ok"

    def test_an_approver_is_kept(self) -> None:
        assert read_resume({"decision": "approved", "approved_by": "staff:kim"})[2] == "staff:kim"
        assert read_resume({"approved": True, "by": "staff:lee"})[2] == "staff:lee"

    def test_no_approver_means_none(self) -> None:
        assert read_resume("approved")[2] is None
