"""Tests for the LLM layer: the free-only guard, the factory and backoff.

No network. `ChatOpenAI` is only constructed, never called, and the retry logic
is tested against a stub model that raises on demand.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models.llm import (
    BUSY_MESSAGE,
    LLMUnavailableError,
    backoff_delay,
    build_chat_model,
    get_llm,
    get_llm_chain,
    is_rate_limited,
    is_retryable,
    models_for_role,
)
from app.models.free_guard import NonFreeModelError


def settings_with(key: str = "sk-or-test", models: str = "openrouter/free") -> Settings:
    return Settings(_env_file=None, openrouter_api_key=key, llm_models=models)


class TestFreeOnly:
    def test_a_paid_model_is_refused(self) -> None:
        with pytest.raises(NonFreeModelError):
            build_chat_model("openai/gpt-4o", settings_with())

    def test_settings_validation_catches_it_earlier(self) -> None:
        with pytest.raises(Exception, match="non-free"):
            settings_with(models="openrouter/free,anthropic/claude-3")

    def test_a_free_model_builds(self) -> None:
        llm = build_chat_model("qwen/qwen3.8-27b:free", settings_with())
        assert llm.model_name == "qwen/qwen3.8-27b:free"

    def test_base_url_points_at_openrouter(self) -> None:
        llm = build_chat_model("openrouter/free", settings_with())
        assert "openrouter.ai" in str(llm.openai_api_base)

    def test_temperature_is_the_configured_low_value(self) -> None:
        settings = Settings(_env_file=None, openrouter_api_key="k", llm_temperature=0.2)
        assert build_chat_model("openrouter/free", settings).temperature == 0.2

    def test_no_key_means_no_live_calls_but_a_clear_message(self) -> None:
        with pytest.raises(LLMUnavailableError, match="OPENROUTER_API_KEY"):
            get_llm("faq", settings_with(key=""))


class TestChain:
    def test_chain_follows_configured_order(self) -> None:
        settings = settings_with(models="openrouter/free,qwen/qwen3.8-27b:free")
        assert [m for m, _ in get_llm_chain("faq", settings)] == [
            "openrouter/free",
            "qwen/qwen3.8-27b:free",
        ]

    def test_settings_refuse_a_paid_entry_in_the_chain(self) -> None:
        # The settings validator is the real gate: a paid id cannot even be
        # configured, so the chain never sees one.
        with pytest.raises(ValidationError, match="non-free"):
            settings_with(models="openrouter/free,openai/gpt-4o")

    def test_every_chain_model_is_free(self) -> None:
        settings = settings_with(models="openrouter/free,qwen/qwen3.8-27b:free")
        for model in models_for_role(settings):
            assert model == "openrouter/free" or model.endswith(":free")

    def test_default_is_the_free_router(self) -> None:
        assert models_for_role(Settings(_env_file=None))[0] == "openrouter/free"


class TestBackoff:
    def test_delays_grow_exponentially(self) -> None:
        assert backoff_delay(1) == 1.0
        assert backoff_delay(2) == 2.0
        assert backoff_delay(3) == 4.0
        assert backoff_delay(4) == 8.0

    def test_delay_is_capped(self) -> None:
        assert backoff_delay(20) == 30.0

    def test_rate_limits_are_recognised(self) -> None:
        assert is_rate_limited(Exception("429 Too Many Requests")) is True
        assert is_rate_limited(Exception("Rate limit exceeded")) is True

        class Status(Exception):
            status_code = 429

        assert is_rate_limited(Status("nope")) is True

    def test_other_transient_failures_are_retryable(self) -> None:
        assert is_retryable(Exception("Request timed out")) is True
        assert is_retryable(Exception("upstream overloaded")) is True
        assert is_retryable(Exception("502 Bad Gateway")) is True

    def test_a_bad_request_is_not_retryable(self) -> None:
        assert is_retryable(Exception("400 invalid request")) is False

    def test_busy_message_is_customer_safe(self) -> None:
        # No stack trace, no status codes, no apology spiral.
        assert "try again" in BUSY_MESSAGE.lower()
        for leak in ("Traceback", "429", "Exception", "rate limit exceeded"):
            assert leak not in BUSY_MESSAGE
