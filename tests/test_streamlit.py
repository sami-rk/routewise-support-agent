"""The Streamlit console, exercised headlessly.

`streamlit.testing.v1.AppTest` runs the script without a browser, which is how a
Streamlit app gets tested. The API is stubbed at the client boundary, so these
tests check the console's own logic — chips, badges, the pending-approval queue —
rather than the agent, which is covered by the graph and API tests.

    python -m pytest tests/test_streamlit.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
sys.path.insert(0, str(FRONTEND))

CHAT_REPLY = {
    "thread_id": "t1",
    "status": "done",
    "response": "The Pro plan is $19 per month. [pricing_plans.md > The plans]",
    "intent": "pricing",
    "intent_conf": 0.9,
    "escalated": False,
    "ticket_id": None,
    "tools_used": [],
    "citations": ["[pricing_plans.md > The plans]"],
    "interrupt": None,
    "trace": [],
}

ESCALATED_REPLY = {
    **CHAT_REPLY,
    "response": "I have passed this to our support team.",
    "intent": "product_info",
    "escalated": True,
    "ticket_id": "tkt_abc123",
}

CONFIRM_REPLY = {
    **CHAT_REPLY,
    "status": "awaiting_confirmation",
    "response": "",
    "interrupt": {
        "mode": "customer_confirm",
        "question": "Shall I cancel your subscription?",
    },
}

APPROVAL_REPLY = {
    **CHAT_REPLY,
    "status": "awaiting_approval",
    "response": "",
    "interrupt": {"mode": "staff_approve", "amount": 49.0, "invoice_id": "inv_barbara_current"},
}

HEALTH = {
    "status": "ok",
    "router": {"backend": "fake", "model": "keywords"},
    "knowledge_base": {"passages": 49},
    "llm_key_present": True,
}


@pytest.fixture
def app(monkeypatch):
    """A running console with the API client stubbed out."""
    import api_client

    calls: list[tuple] = []

    def stub(name, result):
        def handler(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return handler

    monkeypatch.setattr(api_client, "health", stub("health", HEALTH))
    monkeypatch.setattr(api_client, "chat", stub("chat", CHAT_REPLY))
    monkeypatch.setattr(api_client, "confirm", stub("confirm", CHAT_REPLY))
    monkeypatch.setattr(api_client, "decide_approval", stub("decide", {"status": "done"}))
    monkeypatch.setattr(api_client, "metrics", stub("metrics", {
        "conversations": {"threads": 3, "turns": 5, "avg_latency_ms": 1200.0, "p95_latency_ms": 2400.0},
        "outcomes": {"auto_resolution_rate": 0.8, "escalation_rate": 0.2},
        "llm": {"total_requests": 12, "rate_limited": 0, "fallbacks": 0, "requests_per_model": {"openrouter/free": 12}},
        "routing": {"avg_intent_confidence": 0.82, "intent_distribution": {"pricing": 3, "refund": 2}},
    }))
    monkeypatch.setattr(api_client, "tickets", stub("tickets", [
        {"id": "tkt_1", "customer_id": "cus_ada", "subject": "Sync broken", "priority": "high",
         "category": "technical", "status": "open", "created_at": "2026-09-20", "body": "It fails."},
    ]))
    monkeypatch.setattr(api_client, "set_ticket_status", stub("set_status", {"status": "resolved"}))
    monkeypatch.setattr(api_client, "pending_approvals", stub("pending", [
        {"thread_id": "t9", "customer_id": "cus_barbara", "action": "refund", "mode": "staff_approve",
         "status": "awaiting", "payload": '{"amount": 49.0, "invoice_id": "inv_barbara_current", '
                                          '"eligibility": {"reason": "Above the $20 limit."}}'},
    ]))
    monkeypatch.setattr(api_client, "traces", stub("traces", {
        "trace": [{"node": "laya_router", "duration_ms": 2800.0, "model": "laya",
                   "input_summary": "user_input=hi", "error": None}],
        "router_decisions": [{"answers": {"intent": "pricing", "intent_conf": 0.9}}],
    }))

    application = AppTest.from_file(str(FRONTEND / "streamlit_app.py"), default_timeout=60)
    return application, calls


class TestChatPage:
    def test_the_console_starts_without_errors(self, app) -> None:
        application, _ = app
        application.run()
        assert not application.exception

    def test_the_sidebar_reports_the_backend(self, app) -> None:
        application, _ = app
        application.run()
        text = " ".join(c.value for c in application.sidebar.caption)
        assert "backend: ok" in text
        assert "49 passages" in text

    def test_chat_reports_the_intent_and_sources(self, app, monkeypatch) -> None:
        import api_client

        monkeypatch.setattr(api_client, "chat", lambda *a, **k: CHAT_REPLY)
        application, _ = app
        application.run()
        application.chat_input[0].set_value("How much is the Pro plan?").run()
        assert not application.exception
        captions = " ".join(m.value for m in application.caption)
        assert "intent `pricing`" in captions
        assert "pricing_plans.md" in captions

    def test_an_escalation_shows_the_badge_and_ticket(self, app, monkeypatch) -> None:
        import api_client

        monkeypatch.setattr(api_client, "chat", lambda *a, **k: ESCALATED_REPLY)
        application, _ = app
        application.run()
        application.chat_input[0].set_value("I want a human").run()
        assert not application.exception
        errors = " ".join(e.value for e in application.error)
        assert "[ESCALATED]" in errors
        assert "tkt_abc123" in errors

    def test_a_cancellation_asks_before_cancelling(self, app, monkeypatch) -> None:
        import api_client

        monkeypatch.setattr(api_client, "chat", lambda *a, **k: CONFIRM_REPLY)
        application, _ = app
        application.run()
        application.chat_input[0].set_value("Cancel my subscription").run()
        assert not application.exception
        warnings = " ".join(w.value for w in application.warning)
        assert "Shall I cancel your subscription?" in warnings
        assert any("Yes, go ahead" in b.label for b in application.button)

    def test_confirming_calls_the_confirm_endpoint(self, app, monkeypatch) -> None:
        import api_client

        monkeypatch.setattr(api_client, "chat", lambda *a, **k: CONFIRM_REPLY)
        application, calls = app
        application.run()
        application.chat_input[0].set_value("Cancel my subscription").run()
        for button in application.button:
            if "Yes, go ahead" in button.label:
                button.click().run()
                break
        confirmation = next(call for call in calls if call[0] == "confirm")
        # confirm(thread_id, confirmed) — positionally.
        assert confirmation[1][0] == "t1"
        assert confirmation[1][1] is True

    def test_declining_also_confirms(self, app, monkeypatch) -> None:
        import api_client

        monkeypatch.setattr(api_client, "chat", lambda *a, **k: CONFIRM_REPLY)
        application, calls = app
        application.run()
        application.chat_input[0].set_value("Cancel my subscription").run()
        for button in application.button:
            if "No, leave it" in button.label:
                button.click().run()
                break
        confirmation = next(call for call in calls if call[0] == "confirm")
        assert confirmation[1][1] is False


class TestApprovalsPage:
    def test_pending_approvals_are_listed(self, app) -> None:
        application, _ = app
        application.run()
        application.radio[0].set_value("Approvals").run()
        assert not application.exception
        text = " ".join(m.value for m in application.markdown)
        assert "49.0 refund" in text
        assert "inv_barbara_current" in text

    def test_approving_calls_the_decision_endpoint(self, app) -> None:
        application, calls = app
        application.run()
        application.radio[0].set_value("Approvals").run()
        for button in application.button:
            if button.label == "Approve":
                button.click().run()
                break
        decision = next(call for call in calls if call[0] == "decide")
        # decide_approval(thread_id, decision, note) — positionally.
        assert decision[1][1] == "approved"
        assert decision[1][0] == "t9"


class TestTicketsPage:
    def test_tickets_are_tabulated(self, app) -> None:
        application, _ = app
        application.run()
        application.radio[0].set_value("Tickets").run()
        assert not application.exception
        assert application.dataframe

    def test_a_ticket_can_be_opened_and_updated(self, app) -> None:
        application, calls = app
        application.run()
        application.radio[0].set_value("Tickets").run()
        application.selectbox[0].set_value("resolved").run()
        for button in application.button:
            if button.label == "Update":
                button.click().run()
                break
        assert any(call[0] == "set_status" for call in calls)


class TestMetricsPage:
    def test_kpi_cards_are_shown(self, app) -> None:
        application, _ = app
        application.run()
        application.radio[0].set_value("Metrics").run()
        assert not application.exception
        labels = {m.label for m in application.metric}
        assert {"Conversations", "Turns", "Avg latency", "Escalation rate"} <= labels

    def test_intent_distribution_is_charted(self, app) -> None:
        application, _ = app
        application.run()
        application.radio[0].set_value("Metrics").run()
        assert not application.exception
        headings = " ".join(s.value for s in application.subheader)
        assert "Intent distribution" in headings
        assert "Node timeline" in headings
