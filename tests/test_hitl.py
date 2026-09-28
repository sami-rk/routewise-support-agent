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


class TestToolProposedActions:
    """The reliable path: the model calls a proposal tool.

    Free models follow a tool schema but do not reliably emit an `ACTION: {json}`
    line in prose, so the proposal tools are what the graph actually relies on.
    """

    def test_a_refund_proposed_by_tool_is_issued(self, persistent_graph) -> None:
        proposal = {
            "content": "",
            "tool_calls": [
                {
                    "name": "propose_refund",
                    "args": {
                        "invoice_id": "inv_ada_current",
                        "amount": 19.0,
                        "reason": "customer asked",
                    },
                }
            ],
        }
        with ScriptedLLM(
            [proposal, "I have refunded $19.00 to your original payment method."]
        ):
            result = persistent_graph.invoke(
                user_turn("I was charged 5 days ago and want a refund"),
                thread_config("t-tool-refund"),
            )
        assert result.get("__interrupt__") is None
        assert result["action_result"]["ok"] is True
        assert result["action_result"]["amount"] == 19.0

    def test_a_tool_proposed_refund_really_happens(self, persistent_graph) -> None:
        proposal = {
            "content": "",
            "tool_calls": [
                {
                    "name": "propose_refund",
                    "args": {"invoice_id": "inv_ada_current", "amount": 19.0, "reason": "asked"},
                }
            ],
        }
        with ScriptedLLM([proposal, "Refunded."]):
            persistent_graph.invoke(
                user_turn("I was charged 5 days ago and want a refund"),
                thread_config("t-tool-refund2"),
            )
        rows = query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_ada_current'")
        assert len(rows) == 1

    def test_a_tool_proposed_large_refund_pauses(self, persistent_graph) -> None:
        proposal = {
            "content": "",
            "tool_calls": [
                {
                    "name": "propose_refund",
                    "args": {"invoice_id": "inv_barbara_current", "amount": 49.0, "reason": "asked"},
                }
            ],
        }
        with ScriptedLLM([proposal, "It is with our team."]):
            result = persistent_graph.invoke(
                user_turn("refund the 49 dollars please", customer_id="cus_barbara"),
                thread_config("t-tool-big"),
            )
        payload = interrupt_of(result)
        assert payload is not None
        assert payload["mode"] == "staff_approve"
        assert payload["amount"] == 49.0

    def test_a_tool_proposed_cancellation_waits_for_the_customer(self, persistent_graph) -> None:
        proposal = {"content": "", "tool_calls": [{"name": "propose_cancellation", "args": {}}]}
        with ScriptedLLM([proposal, "cancelling"]):
            result = persistent_graph.invoke(
                user_turn("Cancel my subscription"), thread_config("t-tool-cancel")
            )
        assert interrupt_of(result)["mode"] == "customer_confirm"

    def test_a_tool_never_calls_the_gate_without_asking(self, persistent_graph) -> None:
        # A proposal the policy rejects is declined, not sent for approval.
        proposal = {
            "content": "",
            "tool_calls": [
                {
                    "name": "propose_refund",
                    "args": {"invoice_id": "inv_alan_current", "amount": 49.0, "reason": "asked"},
                }
            ],
        }
        with ScriptedLLM([proposal, "That charge is outside the window."]):
            result = persistent_graph.invoke(
                user_turn("refund the 49 dollars", customer_id="cus_alan"),
                thread_config("t-tool-reject"),
            )
        assert query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_alan_current'") == []


class TestActionProposalRetry:
    """When the model explains but never proposes, it is asked once more.

    Observed against the live free models: they would explain the refund policy
    accurately and then not call anything, leaving the customer waiting on an
    action nobody had proposed.
    """

    def test_a_refund_is_proposed_even_if_the_first_reply_only_explains(
        self, persistent_graph
    ) -> None:
        explains = "That charge is 5 days old so it is inside the 14-day window."
        proposal = {
            "content": "",
            "tool_calls": [
                {
                    "name": "propose_refund",
                    "args": {"invoice_id": "inv_ada_current", "amount": 19.0, "reason": "asked"},
                }
            ],
        }
        with ScriptedLLM([explains, proposal, "I have refunded $19.00."]):
            result = persistent_graph.invoke(
                user_turn("I was charged 5 days ago and want a refund"),
                thread_config("t-retry"),
            )
        assert result["action_result"]["ok"] is True
        assert query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_ada_current'")

    def test_the_retry_does_not_run_when_the_model_already_proposed(self, persistent_graph) -> None:
        # Only two replies are scripted: a third call would raise, proving the
        # retry did not happen.
        proposal = {
            "content": "",
            "tool_calls": [
                {
                    "name": "propose_refund",
                    "args": {"invoice_id": "inv_ada_current", "amount": 19.0, "reason": "asked"},
                }
            ],
        }
        with ScriptedLLM([proposal, "Refunded."]):
            result = persistent_graph.invoke(
                user_turn("I was charged 5 days ago and want a refund"),
                thread_config("t-retry-none"),
            )
        assert result["action_result"]["ok"] is True

    def test_no_retry_for_a_question_rather_than_a_request(self, persistent_graph) -> None:
        # A billing *question* is not a request to act, so the retry must not run
        # and must not invent an action. Only two replies are scripted.
        with ScriptedLLM(["You were charged $19.00 on the 10th.", "It was $19.00."]):
            result = persistent_graph.invoke(
                user_turn("how much did you charge me"), thread_config("t-retry-q")
            )
        assert result["pending_action"] is None

    def test_a_refund_outside_the_window_is_not_proposed_on_retry(self, persistent_graph) -> None:
        explains = "That charge is 20 days old, outside the window."
        with ScriptedLLM([explains, "It is outside the refund window."]):
            result = persistent_graph.invoke(
                user_turn("I want my money back", customer_id="cus_alan"),
                thread_config("t-retry-old"),
            )
        assert result.get("action_result") in (None, {"ok": False})
        assert query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_alan_current'") == []


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
