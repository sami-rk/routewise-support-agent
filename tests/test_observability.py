"""Tests for tracing and metrics, against a real database.

Metrics are only worth having if they are true, so these tests run the graph,
then read the numbers back and check they describe what actually happened.
"""

from __future__ import annotations

import json

import pytest

from app.observability.metrics import (
    collect,
    conversation_metrics,
    llm_metrics,
    outcome_metrics,
    percentile,
    refund_metrics,
    routing_metrics,
)
from app.observability.summaries import summarise_input, summarise_output
from app.observability.tracing import get_thread_trace, new_run_id, record_trace, trace_node
from app.db.session import close_connection, execute, query_all, query_one
from tests.conftest import user_turn
from tests.stub_llm import ScriptedLLM


@pytest.fixture
def db(seeded):
    yield seeded
    close_connection()


class TestSummaries:
    def test_input_names_the_useful_fields(self) -> None:
        summary = summarise_input({"user_input": "hello", "intent": "smalltalk", "intent_conf": 0.9})
        assert "user_input=hello" in summary
        assert "intent=smalltalk" in summary

    def test_empty_state_is_labelled(self) -> None:
        assert summarise_input({}) == "(empty state)"

    def test_empty_fields_are_skipped(self) -> None:
        assert summarise_input({"user_input": "hi", "intent": None, "ticket_id": ""}) == "user_input=hi"

    def test_sensitive_fields_are_never_included(self) -> None:
        summary = summarise_input(
            {
                "messages": ["a whole conversation"],
                "retrieved_context": "the whole knowledge base passage",
                "customer_memory": "everything we know about them",
                "user_input": "hello",
            }
        )
        assert "whole conversation" not in summary
        assert "knowledge base" not in summary
        assert "everything we know" not in summary

    def test_output_names_the_response(self) -> None:
        assert "response=hi" in summarise_output({"response": "hi"})

    def test_no_output_is_labelled(self) -> None:
        assert summarise_output({}) == "(no output)"

    def test_long_values_are_clipped(self) -> None:
        assert summarise_output({"response": "x" * 2000}).endswith("...")


class TestTraceNode:
    def test_a_wrapped_node_writes_a_row(self, db) -> None:
        @trace_node("example_node")
        def node(state):
            return {"response": "done"}

        node({"thread_id": "t1", "user_input": "hi", "run_id": new_run_id()})
        rows = query_all("SELECT * FROM traces WHERE node = 'example_node'")
        assert len(rows) == 1
        assert rows[0]["thread_id"] == "t1"
        assert rows[0]["duration_ms"] >= 0

    def test_the_trace_is_returned_in_the_state(self, db) -> None:
        @trace_node("example_node")
        def node(state):
            return {"response": "done"}

        result = node({"thread_id": "t1", "user_input": "hi"})
        assert result["trace"][0]["node"] == "example_node"
        assert result["run_id"]

    def test_a_run_id_is_added_when_missing(self, db) -> None:
        @trace_node("example_node")
        def node(state):
            return {}

        assert node({"thread_id": "t1"})["run_id"]

    def test_a_failing_node_is_traced_with_the_error(self, db) -> None:
        @trace_node("bad_node")
        def node(state):
            raise ValueError("it broke")

        with pytest.raises(ValueError):
            node({"thread_id": "t1", "run_id": new_run_id()})
        row = query_one("SELECT * FROM traces WHERE node = 'bad_node'")
        assert "it broke" in row["error"]

    def test_an_interrupt_is_a_pause_not_an_error(self, db) -> None:
        # `interrupt()` pauses the graph by raising GraphInterrupt. Counting
        # those as failures would make /metrics report an error rate for healthy
        # runs. Raised directly here, because a bare `interrupt()` needs a graph
        # context; `test_interrupts_do_not_count_as_errors` covers the real one.
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Interrupt

        @trace_node("gate")
        def node(state):
            raise GraphInterrupt(Interrupt(value={"mode": "staff_approve"}, id="1"))

        with pytest.raises(GraphInterrupt):
            node({"thread_id": "t1", "run_id": new_run_id()})
        row = query_one("SELECT * FROM traces WHERE node = 'gate'")
        assert row["error"] is None
        assert row["output_summary"] == "interrupted"

    def test_traces_accumulate_rather_than_replace(self, db) -> None:
        @trace_node("example_node")
        def node(state):
            return {}

        first = node({"thread_id": "t1"})
        second = node({"thread_id": "t1", "trace": first["trace"]})
        assert len(second["trace"]) == 2

    def test_thread_trace_is_ordered(self, db) -> None:
        for index in range(3):
            record_trace(
                run_id="r1", thread_id="t1", node=f"n{index}", started_at="2026-01-01 00:00:00",
                duration_ms=index + 1.0,
            )
        assert [row["node"] for row in get_thread_trace("t1")] == ["n0", "n1", "n2"]


class TestMetrics:
    def test_empty_database_gives_zeroes_not_errors(self, db) -> None:
        metrics = collect()
        assert metrics["conversations"]["turns"] == 0
        assert metrics["llm"]["total_requests"] == 0
        assert metrics["refunds"]["refunds"] == 0

    def test_a_real_turn_is_counted(self, db, knowledge_base) -> None:
        from app.agent.graph import build_graph

        graph = build_graph(settings=db)
        with ScriptedLLM(["The Pro plan is $19 per month."]):
            graph.invoke(user_turn("How much is the Pro plan?"))

        conversation = conversation_metrics()
        assert conversation["turns"] >= 1
        assert conversation["avg_latency_ms"] > 0
        assert conversation["p95_latency_ms"] > 0

    def test_routing_distribution_is_reported(self, db, knowledge_base) -> None:
        from app.agent.graph import build_graph

        graph = build_graph(settings=db)
        with ScriptedLLM(["The Pro plan is $19."]):
            graph.invoke(user_turn("How much is the Pro plan?"))

        routing = routing_metrics()
        assert routing["decisions"] >= 1
        assert routing["intent_distribution"].get("pricing", 0) >= 1
        assert routing["avg_intent_confidence"] > 0

    def test_a_router_decision_row_is_logged(self, db, knowledge_base) -> None:
        from app.agent.graph import build_graph

        graph = build_graph(settings=db)
        with ScriptedLLM(["The Pro plan is $19."]):
            graph.invoke(user_turn("How much is the Pro plan?"))

        row = query_one("SELECT * FROM router_decisions ORDER BY id DESC LIMIT 1")
        assert row is not None
        answers = json.loads(row["answers_json"])
        assert answers["intent"] == "pricing"
        probs = json.loads(row["probs_json"])
        # The full intent distribution is kept, not just the winner, so the
        # routing eval can compute calibration.
        assert probs["intent"]["pricing"] == pytest.approx(0.9)
        assert "injection" in probs

    def test_escalation_is_counted(self, db, knowledge_base) -> None:
        from app.agent.graph import build_graph

        graph = build_graph(settings=db)
        with ScriptedLLM(["Handing over."]):
            graph.invoke(user_turn("I want a human"))

        outcomes = outcome_metrics()
        assert outcomes["escalated_runs"] >= 1
        assert outcomes["escalation_rate"] > 0
        assert outcomes["tickets"] >= 1

    def test_interrupts_do_not_count_as_errors(self, db, knowledge_base) -> None:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.types import Command

        from app.agent.graph import build_graph, thread_config
        from tests.stub_llm import ScriptedLLM

        proposal = {"content": "", "tool_calls": [{"name": "propose_cancellation", "args": {}}]}
        with SqliteSaver.from_conn_string(str(db.checkpoint_db_path)) as saver:
            graph = build_graph(checkpointer=saver, settings=db)
            with ScriptedLLM([proposal, "cancelling", "cancelled"]):
                graph.invoke(
                    user_turn("Cancel my subscription", thread_id="t-int"),
                    thread_config("t-int"),
                )
                graph.invoke(Command(resume="approved"), thread_config("t-int"))

        assert conversation_metrics()["errors"] == 0
        assert outcome_metrics()["escalated_runs"] == 0

    def test_llm_requests_are_counted_per_model(self, db) -> None:
        execute(
            "INSERT INTO llm_calls (role, model, attempt, status) VALUES ('faq', 'openrouter/free', 1, 'ok')"
        )
        execute(
            "INSERT INTO llm_calls (role, model, attempt, status, error) "
            "VALUES ('faq', 'openrouter/free', 1, 'rate_limited', '429')"
        )
        metrics = llm_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["successful"] == 1
        assert metrics["rate_limited"] == 1
        assert metrics["requests_per_model"]["openrouter/free"] == 2

    def test_refund_metrics_split_auto_from_staff(self, db) -> None:
        execute(
            "INSERT INTO refunds (id, invoice_id, amount, approved_by) "
            "VALUES ('rf1', 'inv_ada_current', 10.0, 'automatic')"
        )
        execute(
            "INSERT INTO refunds (id, invoice_id, amount, approved_by) "
            "VALUES ('rf2', 'inv_ada_current', 5.0, 'staff:kim')"
        )
        metrics = refund_metrics()
        assert metrics["refunds"] == 2
        assert metrics["auto_approved"] == 1
        assert metrics["staff_approved"] == 1
        assert metrics["total_refunded"] == 15.0

    def test_pending_approvals_are_counted(self, db) -> None:
        execute(
            "INSERT INTO pending_actions (thread_id, customer_id, action, payload, mode) "
            "VALUES ('t1', 'cus_ada', 'refund', '{}', 'staff_approve')"
        )
        assert refund_metrics()["pending"] == 1


class TestPercentile:
    def test_median_of_an_odd_list(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_p95_picks_an_observed_value(self) -> None:
        values = [float(n) for n in range(1, 101)]
        assert percentile(values, 0.95) in values

    def test_empty_is_zero(self) -> None:
        assert percentile([], 0.95) == 0.0

    def test_single_value(self) -> None:
        assert percentile([7.0], 0.95) == 7.0
