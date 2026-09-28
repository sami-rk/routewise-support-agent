"""Tests for the fallback runner.

The models are stubs that succeed or raise on demand, so every branch —
first model works, first is rate limited, everything fails — is exercised with
no network and no sleeping.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.models import llm_runner
from app.models.llm import BUSY_MESSAGE, LLMUnavailableError
from app.models.llm_runner import invoke_with_fallback


class StubResponse:
    def __init__(self, text: str = "answer") -> None:
        self.text = text
        self.content = text
        self.usage_metadata = {"input_tokens": 11, "output_tokens": 7}
        self.response_metadata = {}


class StubModel:
    """Raises whatever it was given, for the first `failures` calls."""

    def __init__(self, failures: list[Exception] | None = None, text: str = "answer") -> None:
        self.failures = list(failures or [])
        self.text = text
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return StubResponse(self.text)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Backoff is real in production and instant in tests."""
    monkeypatch.setattr(llm_runner, "sleep_for", lambda seconds: None)
    monkeypatch.setattr(llm_runner, "MAX_ATTEMPTS", 2)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="sk-or-test",
        llm_models="openrouter/free,qwen/qwen3.8-27b:free",
    )


def use_models(monkeypatch, models: dict[str, StubModel]) -> None:
    monkeypatch.setattr(llm_runner, "get_llm_chain", lambda *a, **k: list(models.items()))


class TestHappyPath:
    def test_first_model_answers(self, monkeypatch, settings) -> None:
        first = StubModel(text="from the first model")
        use_models(monkeypatch, {"openrouter/free": first})
        result = invoke_with_fallback("hi", "faq", settings, record=False)
        assert result.text == "from the first model"
        assert first.calls == 1

    def test_tokens_are_read_from_the_response(self, monkeypatch, settings) -> None:
        use_models(monkeypatch, {"openrouter/free": StubModel()})
        result = invoke_with_fallback("hi", "faq", settings, record=False)
        assert result.usage_metadata["input_tokens"] == 11


class TestFallback:
    def test_a_retryable_failure_moves_to_the_next_model(self, monkeypatch, settings) -> None:
        first = StubModel([Exception("429 Too Many Requests"), Exception("429 Too Many Requests")])
        second = StubModel(text="from the fallback")
        use_models(monkeypatch, {"openrouter/free": first, "qwen/qwen3.8-27b:free": second})

        result = invoke_with_fallback("hi", "faq", settings, record=False)
        assert result.text == "from the fallback"
        assert first.calls == 2, "the first model should be retried once"
        assert second.calls == 1

    def test_a_transient_error_is_retried_on_the_same_model(self, monkeypatch, settings) -> None:
        flaky = StubModel([Exception("upstream overloaded")], text="recovered")
        use_models(monkeypatch, {"openrouter/free": flaky})
        assert invoke_with_fallback("hi", "faq", settings, record=False).text == "recovered"
        assert flaky.calls == 2

    def test_an_unretryable_error_moves_on_without_waiting(self, monkeypatch, settings) -> None:
        first = StubModel([Exception("400 invalid request")])
        second = StubModel(text="second model")
        use_models(monkeypatch, {"openrouter/free": first, "qwen/qwen3.8-27b:free": second})
        assert invoke_with_fallback("hi", "faq", settings, record=False).text == "second model"
        assert first.calls == 1, "a 400 should not be retried"

    def test_every_model_is_tried(self, monkeypatch, settings) -> None:
        models = {
            "openrouter/free": StubModel([Exception("500 Server Error")] * 2),
            "qwen/qwen3.8-27b:free": StubModel([Exception("500 Server Error")] * 2),
        }
        use_models(monkeypatch, models)
        with pytest.raises(LLMUnavailableError):
            invoke_with_fallback("hi", "faq", settings, record=False, max_steps=2)
        assert all(m.calls == 2 for m in models.values())


class TestFailure:
    def test_all_models_failing_gives_a_friendly_message(self, monkeypatch, settings) -> None:
        use_models(
            monkeypatch,
            {"openrouter/free": StubModel([Exception("429 rate limit")] * 2)},
        )
        with pytest.raises(LLMUnavailableError) as excinfo:
            invoke_with_fallback("hi", "faq", settings, record=False)
        # The customer never sees a stack trace or a status code.
        assert str(excinfo.value) == BUSY_MESSAGE
        assert "429" not in str(excinfo.value)
        assert "Traceback" not in str(excinfo.value)

    def test_the_original_error_is_kept_as_the_cause(self, monkeypatch, settings) -> None:
        use_models(monkeypatch, {"openrouter/free": StubModel([Exception("503 overloaded")] * 2)})
        with pytest.raises(LLMUnavailableError) as excinfo:
            invoke_with_fallback("hi", "faq", settings, record=False)
        assert isinstance(excinfo.value.__cause__, Exception)
        assert "503" in str(excinfo.value.__cause__)

    def test_no_models_configured_gives_the_friendly_message(self, monkeypatch, settings) -> None:
        monkeypatch.setattr(llm_runner, "get_llm_chain", lambda *a, **k: [])
        with pytest.raises(LLMUnavailableError) as excinfo:
            invoke_with_fallback("hi", "faq", settings, record=False)
        assert str(excinfo.value) == BUSY_MESSAGE


class TestBudget:
    def test_the_number_of_requests_per_turn_is_bounded(self, monkeypatch, settings) -> None:
        # The whole point of the free tier: a turn cannot make many requests.
        models = {
            "openrouter/free": StubModel([Exception("429")] * 2),
            "qwen/qwen3.8-27b:free": StubModel([Exception("429")] * 2),
        }
        use_models(monkeypatch, models)
        with pytest.raises(LLMUnavailableError):
            invoke_with_fallback("hi", "faq", settings, record=False, max_steps=2)
        assert sum(m.calls for m in models.values()) == 4  # 2 models x 2 attempts, no more
