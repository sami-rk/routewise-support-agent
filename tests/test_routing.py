"""Tests for the conditional edge functions.

These are pure functions, so the whole routing table is tested here without a
graph, a model or a database. The graph-level paths are tested separately in
`test_graph.py`.
"""

from __future__ import annotations

import pytest

from app.agent.routing import (
    after_agent,
    after_approval,
    after_billing,
    is_injection,
    is_low_confidence,
    needs_human,
    route_by_intent,
)
from app.config import Settings


def state(**kwargs):
    base = {"intent": "product_info", "intent_conf": 0.9}
    base.update(kwargs)
    return base


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, router_min_conf=0.55)


class TestRouteByIntent:
    @pytest.mark.parametrize(
        ("intent", "expected"),
        [
            ("billing", "billing_agent"),
            ("refund", "billing_agent"),
            ("cancellation", "billing_agent"),
            ("technical", "technical_agent"),
            ("product_info", "faq_agent"),
            ("pricing", "faq_agent"),
            ("account", "faq_agent"),
            ("smalltalk", "faq_agent"),
        ],
    )
    def test_each_intent_reaches_its_agent(
        self, intent: str, expected: str, settings: Settings
    ) -> None:
        assert route_by_intent(state(intent=intent), settings) == expected

    def test_unknown_intent_falls_back_to_faq(self, settings: Settings) -> None:
        assert route_by_intent(state(intent="something_new"), settings) == "faq_agent"

    def test_missing_intent_falls_back_to_faq(self, settings: Settings) -> None:
        assert route_by_intent({"intent_conf": 0.9}, settings) == "faq_agent"


class TestEscalation:
    def test_needs_human_wins(self, settings: Settings) -> None:
        assert route_by_intent(state(needs_human=True, intent="refund"), settings) == "escalate"

    def test_angry_and_urgent_escalates(self, settings: Settings) -> None:
        turn = state(frustrated=True, urgency=0.8, intent="billing")
        assert needs_human(turn) is True
        assert route_by_intent(turn, settings) == "escalate"

    def test_angry_but_calm_does_not_escalate(self, settings: Settings) -> None:
        turn = state(frustrated=True, urgency=0.2, intent="billing")
        assert needs_human(turn) is False
        assert route_by_intent(turn, settings) == "billing_agent"

    def test_urgent_but_calm_does_not_escalate(self, settings: Settings) -> None:
        assert needs_human(state(frustrated=False, urgency=1.0, intent="technical")) is False

    def test_legacy_escalate_flag_still_honoured(self, settings: Settings) -> None:
        assert needs_human(state(escalate=True)) is True

    def test_agent_requested_handoff_escalates(self, settings: Settings) -> None:
        turn = state(escalated_reason="agent_requested")
        assert needs_human(turn) is True


class TestClarification:
    def test_low_confidence_asks_once(self, settings: Settings) -> None:
        assert route_by_intent(state(intent_conf=0.3), settings) == "clarify"

    def test_high_confidence_does_not_ask(self, settings: Settings) -> None:
        assert route_by_intent(state(intent_conf=0.9), settings) != "clarify"

    def test_the_threshold_is_the_configured_one(self) -> None:
        strict = Settings(_env_file=None, router_min_conf=0.95)
        assert is_low_confidence(state(intent_conf=0.9), strict) is True
        assert is_low_confidence(state(intent_conf=0.9), Settings(_env_file=None)) is False

    def test_still_unsure_after_clarifying_escalates(self, settings: Settings) -> None:
        turn = state(intent_conf=0.3, awaiting_clarification=True)
        assert route_by_intent(turn, settings) == "escalate"

    def test_a_clear_answer_after_clarifying_proceeds(self, settings: Settings) -> None:
        turn = state(intent_conf=0.9, awaiting_clarification=True, intent="pricing")
        assert route_by_intent(turn, settings) == "faq_agent"


class TestInjection:
    def test_injection_answers_directly(self, settings: Settings) -> None:
        turn = state(injection=True, intent="refund")
        assert is_injection(turn) is True
        assert route_by_intent(turn, settings) == "respond"

    def test_injection_beats_escalation(self, settings: Settings) -> None:
        # An injection attempt gets the safe reply, not a handoff to a person.
        turn = state(injection=True, needs_human=True)
        assert route_by_intent(turn, settings) == "respond"

    def test_a_probability_above_the_threshold_counts(self, settings: Settings) -> None:
        assert route_by_intent(state(injection=0.9), settings) == "respond"
        assert route_by_intent(state(injection=0.5), settings) != "respond"

    def test_no_injection_signal(self, settings: Settings) -> None:
        assert is_injection(state(injection=0.0)) is False
        assert is_injection(state()) is False


class TestAfterAgent:
    def test_plain_answer_responds(self) -> None:
        assert after_agent(state()) == "respond"

    def test_ticket_request_creates_one(self) -> None:
        assert after_agent(state(wants_ticket=True)) == "create_ticket"

    def test_existing_ticket_is_not_duplicated(self) -> None:
        assert after_agent(state(wants_ticket=True, ticket_id="tkt_1")) == "create_ticket"


class TestAfterBilling:
    def test_no_action_responds(self) -> None:
        assert after_billing(state()) == "respond"

    def test_a_proposed_action_goes_to_the_gate(self) -> None:
        assert after_billing(state(pending_action={"type": "refund"})) == "approval_gate"


class TestAfterApproval:
    def test_approved_runs_the_action(self) -> None:
        turn = state(approval="approved", pending_action={"type": "refund"})
        assert after_approval(turn) == "execute_action"

    def test_rejected_does_not(self) -> None:
        turn = state(approval="rejected", pending_action={"type": "refund"})
        assert after_approval(turn) == "respond"

    def test_unanswered_does_not(self) -> None:
        turn = state(pending_action={"type": "refund"})
        assert after_approval(turn) == "respond"

    def test_approved_without_an_action_does_not(self) -> None:
        assert after_approval(state(approval="approved")) == "respond"
