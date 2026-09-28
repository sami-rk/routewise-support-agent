"""Tests for the Laya question definitions and the LayaRouter's answer parsing.

The question definitions are checked against the real `laya` package whenever
it is importable, because a malformed question fails deep inside the model
rather than at the call site. The parsing tests never import torch: they feed
`LayaRouter` a stub agent, so the whole file runs offline.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.models.router import (
    INJECTION_THRESHOLD,
    INTENTS,
    NEEDS_HUMAN_THRESHOLD,
    QUESTIONS,
    URGENCY_LEVELS,
    LayaRouter,
    RouterModel,
    resolve_device,
)

try:  # pragma: no cover - depends on the environment
    import laya as _laya  # noqa: F401

    HAS_LAYA = True
except Exception:  # pragma: no cover
    HAS_LAYA = False


class TestQuestions:
    def test_all_five_questions_are_asked(self) -> None:
        assert set(QUESTIONS) == {
            "intent",
            "urgency",
            "frustrated",
            "needs_human",
            "injection",
        }

    def test_question_types_match_the_spec(self) -> None:
        assert QUESTIONS["intent"]["type"] == "choice"
        assert QUESTIONS["urgency"]["type"] == "score"
        assert QUESTIONS["frustrated"]["type"] == "noul"
        assert QUESTIONS["needs_human"]["type"] == "noul"
        assert QUESTIONS["injection"]["type"] == "noul"

    def test_every_question_has_instructions(self) -> None:
        for qid, question in QUESTIONS.items():
            assert question["instructions"].strip(), qid

    def test_intent_criteria_cover_every_intent(self) -> None:
        assert tuple(QUESTIONS["intent"]["criteria"]) == INTENTS

    def test_intent_criteria_describe_each_option(self) -> None:
        for label, description in QUESTIONS["intent"]["criteria"].items():
            assert label and description.strip(), label

    def test_urgency_scale_is_ordered_low_to_high(self) -> None:
        assert QUESTIONS["urgency"]["criteria"] == list(URGENCY_LEVELS)

    def test_thresholds_are_in_the_zero_to_one_range(self) -> None:
        assert 0.0 < NEEDS_HUMAN_THRESHOLD <= 1.0
        assert 0.0 < INJECTION_THRESHOLD <= 1.0

    @pytest.mark.skipif(not HAS_LAYA, reason="laya is not installed")
    def test_questions_pass_layas_own_validation(self) -> None:
        # `Agent._check_question` is the validator that names a malformed
        # question, so ask it directly rather than discovering the problem
        # inside a forward pass.
        from laya.agent import Agent

        for qid, question in QUESTIONS.items():
            Agent._check_question(qid, question)

    @pytest.mark.skipif(not HAS_LAYA, reason="laya is not installed")
    def test_intent_renders_eight_options(self) -> None:
        from laya.agent import Agent
        from laya.common import render_options

        internal = Agent._to_internal(QUESTIONS["intent"])
        assert len(render_options(internal)) == len(INTENTS) == 8

    @pytest.mark.skipif(not HAS_LAYA, reason="laya is not installed")
    def test_urgency_renders_five_levels(self) -> None:
        from laya.agent import Agent
        from laya.common import render_options

        internal = Agent._to_internal(QUESTIONS["urgency"])
        assert len(render_options(internal)) == len(URGENCY_LEVELS) == 5


def make_router(payload: dict[str, Any]) -> LayaRouter:
    """A LayaRouter whose agent returns `payload`, without importing laya."""
    router = LayaRouter()
    router._agent = object()  # marks it as loaded

    class StubAgent:
        def system_one(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
            return json.loads(json.dumps(payload))  # deep copy, as laya does

    router._agent = StubAgent()
    return router


def sample_payload() -> dict[str, Any]:
    return {
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "refund",
                "probabilities": {
                    "product_info": 0.02,
                    "pricing": 0.03,
                    "billing": 0.15,
                    "refund": 0.70,
                    "cancellation": 0.04,
                    "technical": 0.03,
                    "account": 0.02,
                    "smalltalk": 0.01,
                },
                "confidence": 0.62,
                "answer_confidence": 0.70,
            },
            "urgency": {
                "type": "score",
                "score": 2.4,
                "legend": {str(i): label for i, label in enumerate(URGENCY_LEVELS)},
                "probabilities": {"0": 0.01, "1": 0.05, "2": 0.55, "3": 0.3, "4": 0.09},
            },
            "frustrated": {"type": "noul", "noul": 0.12, "confidence": 0.88},
            "needs_human": {"type": "noul", "noul": 0.05, "confidence": 0.95},
            "injection": {"type": "noul", "noul": 0.02, "confidence": 0.98},
        },
        "routing": {"model": "english", "reason": "english script detected"},
    }


class TestLayaRouterParsing:
    def test_satisfies_the_router_interface(self) -> None:
        assert isinstance(make_router(sample_payload()), RouterModel)

    def test_reads_the_chosen_intent(self) -> None:
        decision = make_router(sample_payload()).predict("I want a refund")
        assert decision.intent == "refund"

    def test_confidence_is_the_top_probability_not_the_entropy_score(self) -> None:
        # The payload's `confidence` is 0.62 (normalised entropy) while the top
        # probability is 0.70; the gate must use the latter.
        decision = make_router(sample_payload()).predict("I want a refund")
        assert decision.intent_conf == pytest.approx(0.70)

    def test_keeps_the_whole_intent_distribution(self) -> None:
        decision = make_router(sample_payload()).predict("I want a refund")
        assert set(decision.intent_probs) == set(INTENTS)
        assert sum(decision.intent_probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_maps_urgency_onto_the_scale(self) -> None:
        decision = make_router(sample_payload()).predict("I want a refund")
        assert decision.urgency == pytest.approx(0.6)
        assert decision.urgency_band == "medium"

    def test_noul_questions_become_flags_with_confidence(self) -> None:
        decision = make_router(sample_payload()).predict("I want a refund")
        assert decision.frustrated is False
        assert decision.frustrated_conf == pytest.approx(0.12)
        assert decision.needs_human is False
        assert decision.needs_human_conf == pytest.approx(0.05)
        assert decision.injection is False
        assert decision.injection_conf == pytest.approx(0.02)

    def test_high_noul_probability_sets_the_flag(self) -> None:
        payload = sample_payload()
        payload["answers"]["needs_human"]["noul"] = 0.95
        decision = make_router(payload).predict("I want a human")
        assert decision.needs_human is True
        assert decision.wants_human is True

    def test_records_the_checkpoint_and_backend(self) -> None:
        decision = make_router(sample_payload()).predict("hello")
        assert decision.model == "english"
        assert decision.backend == "laya"

    def test_state_is_truncated_before_it_is_sent(self) -> None:
        seen: list[str] = []

        router = LayaRouter()
        router._agent = None

        class Recorder:
            def system_one(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
                seen.append(state)
                return sample_payload()

        router._agent = Recorder()
        router.predict("q" * 5000)
        assert len(seen[0]) <= 1200

    def test_falls_back_to_predict_when_system_one_is_absent(self) -> None:
        router = LayaRouter()

        class OldAgent:
            def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
                return sample_payload()

        router._agent = OldAgent()
        assert router.predict("hello").intent == "refund"

    def test_unknown_intent_label_is_dropped(self) -> None:
        payload = sample_payload()
        payload["answers"]["intent"]["choice"] = "not_an_intent"
        decision = make_router(payload).predict("hello")
        assert decision.intent is None

    def test_missing_answers_do_not_crash(self) -> None:
        decision = make_router({"answers": {}}).predict("hello")
        assert decision.intent is None
        assert decision.intent_conf is None
        assert decision.urgency == 0.0
        assert decision.wants_human is False

    def test_raw_payload_is_kept_for_the_trace(self) -> None:
        decision = make_router(sample_payload()).predict("hello")
        assert "answers" in decision.raw
        assert decision.raw["routing"]["model"] == "english"

    def test_health_reports_whether_it_is_loaded(self) -> None:
        assert LayaRouter().health()["loaded"] is False
        assert make_router(sample_payload()).health()["loaded"] is True

    def test_agent_without_either_method_raises_a_clear_error(self) -> None:
        router = LayaRouter()
        router._agent = object()
        with pytest.raises(RuntimeError, match="system_one"):
            router.predict("hello")


def test_resolve_device_defaults_to_cpu_without_cuda() -> None:
    # This box has no GPU, and the latency budget is met on CPU.
    assert resolve_device("auto") in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"
