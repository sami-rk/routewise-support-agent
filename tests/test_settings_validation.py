"""The startup settings validator.

`get_settings` is lazy, so `validate_settings` is what makes a bad configuration
fail when a process boots rather than on its first request.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings, reset_settings_cache, validate_settings


def test_a_good_environment_validates() -> None:
    assert validate_settings(Settings(_env_file=None, llm_models=["openrouter/free"]))


def test_validation_returns_the_settings_given_to_it() -> None:
    settings = Settings(_env_file=None)
    assert validate_settings(settings) is settings


def test_a_paid_model_cannot_be_validated(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODELS", "openai/gpt-4o")
    reset_settings_cache()
    try:
        with pytest.raises(ValidationError, match="non-free"):
            validate_settings()
    finally:
        reset_settings_cache()


def test_a_free_model_validates(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODELS", "openrouter/free,some/model:free")
    reset_settings_cache()
    try:
        assert validate_settings().llm_models == ["openrouter/free", "some/model:free"]
    finally:
        reset_settings_cache()


def test_a_bad_kb_top_k_cannot_be_validated(monkeypatch) -> None:
    monkeypatch.setenv("KB_TOP_K", "0")
    reset_settings_cache()
    try:
        with pytest.raises(ValidationError):
            validate_settings()
    finally:
        reset_settings_cache()


def test_the_cache_is_cleared_between_reads(monkeypatch) -> None:
    monkeypatch.setenv("ROUTER_MIN_CONF", "0.9")
    reset_settings_cache()
    try:
        assert get_settings().router_min_conf == 0.9
        monkeypatch.setenv("ROUTER_MIN_CONF", "0.2")
        assert get_settings().router_min_conf == 0.9, "still cached"
        reset_settings_cache()
        assert get_settings().router_min_conf == 0.2
    finally:
        reset_settings_cache()
