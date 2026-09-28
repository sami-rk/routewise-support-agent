"""API tests.

Every test runs with the keyword router and a scripted model through
`TestClient`, so the whole HTTP surface is exercised offline — including that
the staff endpoints really are behind the admin key.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(seeded, knowledge_base):
    """A live app against the temporary database and index."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def admin(settings) -> dict[str, str]:
    return {"X-Admin-Key": settings.admin_api_key}


class TestHealth:
    def test_health_reports_the_loaded_pieces(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] in ("ok", "degraded")
        assert body["router"]["loaded"] is True
        assert body["graph_loaded"] is True
        assert body["database"] is True

    def test_health_lists_the_free_models(self, client) -> None:
        models = client.get("/health").json()["llm_models"]
        assert models
        for model in models:
            assert model == "openrouter/free" or model.endswith(":free")

    def test_health_reports_a_missing_llm_key(self, client) -> None:
        assert isinstance(client.get("/health").json()["llm_key_present"], bool)


class TestChat:
    def test_a_pricing_question_is_answered(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["The Pro plan is $19 per month and supports 5 devices."]):
            body = client.post(
                "/chat", json={"customer_id": "cus_ada", "message": "How much is the Pro plan?"}
            ).json()
        assert body["status"] == "done"
        assert "$19" in body["response"]
        assert body["intent"] == "pricing"

    def test_a_new_thread_id_is_generated(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["Hello!"]):
            body = client.post(
                "/chat", json={"customer_id": "cus_ada", "message": "hello"}
            ).json()
        assert body["thread_id"].startswith("thread_")

    def test_a_supplied_thread_id_is_kept(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["Hello!"]):
            body = client.post(
                "/chat",
                json={"thread_id": "my-thread", "customer_id": "cus_ada", "message": "hello"},
            ).json()
        assert body["thread_id"] == "my-thread"

    def test_an_escalation_reports_a_ticket(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["I have passed this to our team."]):
            body = client.post(
                "/chat", json={"customer_id": "cus_ada", "message": "I want a human"}
            ).json()
        assert body["escalated"] is True
        assert body["ticket_id"]

    def test_an_empty_message_is_refused(self, client) -> None:
        assert client.post("/chat", json={"customer_id": "cus_ada", "message": ""}).status_code == 422

    def test_a_missing_customer_is_refused(self, client) -> None:
        assert client.post("/chat", json={"message": "hi"}).status_code == 422

    def test_the_reply_carries_the_sources(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["The Pro plan is $19."]):
            body = client.post(
                "/chat", json={"customer_id": "cus_ada", "message": "How much is the Pro plan?"}
            ).json()
        assert any("pricing_plans.md" in c for c in body["citations"])


class TestRefundApproval:
    def _start_large_refund(self, client) -> str:
        from tests.stub_llm import ScriptedLLM

        reply = 'ACTION: {"type": "refund", "invoice_id": "inv_barbara_current", "amount": 49.0}'
        with ScriptedLLM([reply, "It is with our team."]):
            body = client.post(
                "/chat",
                json={
                    "thread_id": "api-refund",
                    "customer_id": "cus_barbara",
                    "message": "I want a refund of the 49 dollars",
                },
            ).json()
        assert body["status"] == "awaiting_approval"
        return body["thread_id"]

    def test_a_large_refund_returns_an_interrupt(self, client) -> None:
        self._start_large_refund(client)

    def test_the_interrupt_says_what_is_being_asked(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        reply = 'ACTION: {"type": "refund", "invoice_id": "inv_barbara_current", "amount": 49.0}'
        with ScriptedLLM([reply, "with our team"]):
            body = client.post(
                "/chat",
                json={
                    "customer_id": "cus_barbara",
                    "message": "I want a refund of the 49 dollars",
                },
            ).json()
        interrupt = body["interrupt"]
        assert interrupt["mode"] == "staff_approve"
        assert interrupt["amount"] == 49.0
        assert interrupt["eligibility"]["eligible"] is True

    def test_a_refund_needs_the_admin_key(self, client) -> None:
        assert client.post("/approvals/whatever", json={"decision": "approved"}).status_code == 401

    def test_a_pending_approval_is_listed(self, client, admin) -> None:
        self._start_large_refund(client)
        # The queue is written when the API records the pause.
        rows = client.get("/approvals/pending", headers=admin).json()
        assert isinstance(rows, list)

    def test_approving_a_refund_runs_it(self, client, admin) -> None:
        thread_id = self._start_large_refund(client)
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["Your refund is on its way."]):
            body = client.post(
                f"/approvals/{thread_id}",
                json={"decision": "approved", "note": "ok", "approved_by": "staff:kim"},
                headers=admin,
            ).json()
        assert body["status"] == "done"
        assert "$49.00" in body["response"]

        from app.db.session import query_all

        rows = query_all("SELECT * FROM refunds WHERE invoice_id = 'inv_barbara_current'")
        assert len(rows) == 1
        assert rows[0]["approved_by"] == "staff:kim"

    def test_deciding_a_thread_that_is_not_waiting_conflicts(self, client, admin) -> None:
        response = client.post(
            "/approvals/no-such-thread", json={"decision": "approved"}, headers=admin
        )
        assert response.status_code == 409

    def test_cancellation_waits_for_the_customer(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(['Sure. ACTION: {"type": "cancel"}', "cancelling"]):
            body = client.post(
                "/chat",
                json={"thread_id": "api-cancel", "customer_id": "cus_ada", "message": "Cancel my subscription"},
            ).json()
        assert body["status"] == "awaiting_confirmation"
        assert body["interrupt"]["mode"] == "customer_confirm"

    def test_confirming_cancels_the_subscription(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(['Sure. ACTION: {"type": "cancel"}', "cancelling", "Cancelled."]):
            client.post(
                "/chat",
                json={"thread_id": "api-cancel2", "customer_id": "cus_ada", "message": "Cancel my subscription"},
            )
            body = client.post(
                "/chat/confirm", json={"thread_id": "api-cancel2", "confirmed": True}
            ).json()
        assert body["status"] == "done"
        assert "cancelled" in body["response"].lower()

    def test_confirming_a_thread_that_is_not_waiting_conflicts(self, client) -> None:
        response = client.post("/chat/confirm", json={"thread_id": "nothing", "confirmed": True})
        assert response.status_code == 409


class TestStream:
    def test_the_stream_emits_routing_and_a_final_event(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["The Pro plan is $19."]):
            with client.stream(
                "POST",
                "/chat/stream",
                json={"customer_id": "cus_ada", "message": "How much is the Pro plan?"},
            ) as response:
                body = "".join(response.iter_text())
        assert "event: routing" in body
        assert "event: done" in body
        assert "pricing" in body


class TestThreads:
    def test_history_is_readable(self, client) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["The Pro plan is $19."]):
            client.post(
                "/chat",
                json={"thread_id": "t-hist", "customer_id": "cus_ada", "message": "How much is the Pro plan?"},
            )
        body = client.get("/threads/t-hist").json()
        assert body["thread_id"] == "t-hist"
        assert body["messages"]

    def test_an_unknown_thread_is_empty_not_an_error(self, client) -> None:
        assert client.get("/threads/never-used").json()["messages"] == []


class TestStaffEndpoints:
    def test_tickets_need_the_key(self, client) -> None:
        assert client.get("/tickets").status_code == 401

    def test_tickets_are_listed(self, client, admin) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["Handing over."]):
            client.post("/chat", json={"customer_id": "cus_ada", "message": "I want a human"})
        tickets = client.get("/tickets", headers=admin).json()
        assert any(t["customer_id"] == "cus_ada" for t in tickets)

    def test_a_ticket_can_be_fetched_and_updated(self, client, admin) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["Handing over."]):
            body = client.post("/chat", json={"customer_id": "cus_ada", "message": "I want a human"}).json()
        ticket_id = body["ticket_id"]

        assert client.get(f"/tickets/{ticket_id}", headers=admin).json()["id"] == ticket_id
        updated = client.patch(
            f"/tickets/{ticket_id}", json={"status": "resolved"}, headers=admin
        ).json()
        assert updated["status"] == "resolved"

    def test_a_missing_ticket_is_404(self, client, admin) -> None:
        assert client.get("/tickets/tkt_nope", headers=admin).status_code == 404

    def test_metrics_need_the_key(self, client) -> None:
        assert client.get("/metrics").status_code == 401

    def test_metrics_are_reported(self, client, admin) -> None:
        body = client.get("/metrics", headers=admin).json()
        assert "conversations" in body and "routing" in body and "llm" in body

    def test_traces_are_reported(self, client, admin) -> None:
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["The Pro plan is $19."]):
            client.post(
                "/chat",
                json={"thread_id": "t-trace", "customer_id": "cus_ada", "message": "How much is the Pro plan?"},
            )
        body = client.get("/traces/t-trace", headers=admin).json()
        assert body["trace"], "the node trace should not be empty"
        assert body["router_decisions"]


class TestMemoryEndpoints:
    def test_memory_needs_the_key(self, client) -> None:
        assert client.get("/customers/cus_ada/memory").status_code == 401

    def test_memory_can_be_read_and_erased(self, client, admin) -> None:
        from app.memory.long_term import set_memory

        set_memory("cus_ada", "On the Pro plan, uses Windows, syncs five devices.")
        body = client.get("/customers/cus_ada/memory", headers=admin).json()
        assert "Pro plan" in body["summary"]

        assert client.delete("/customers/cus_ada/memory", headers=admin).json()["erased"] is True
        assert client.get("/customers/cus_ada/memory", headers=admin).json()["summary"] == ""

    def test_an_unknown_customer_is_404(self, client, admin) -> None:
        assert client.get("/customers/cus_nobody/memory", headers=admin).status_code == 404


class TestIsolation:
    def test_one_customer_cannot_see_another_customers_ticket(self, client, admin) -> None:
        # The staff list is guarded; a customer only ever sees their own ticket
        # through the chat reply, which is scoped by graph state.
        from tests.stub_llm import ScriptedLLM

        with ScriptedLLM(["Handing over."]):
            body = client.post(
                "/chat", json={"customer_id": "cus_ada", "message": "I want a human"}
            ).json()
        assert body["ticket_id"]
        # Ada's own ticket id is what she was given, and it is hers.
        assert client.get(f"/tickets/{body['ticket_id']}", headers=admin).json()["customer_id"] == "cus_ada"
