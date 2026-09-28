"""Tests for the free-tier model guard."""

from __future__ import annotations

import pytest

from app.config import is_free_model
from app.models.free_guard import NonFreeModelError, assert_free_model, free_model_candidates


@pytest.mark.parametrize(
    "model_id",
    ["openrouter/free", "qwen/qwen3.8-27b:free", "nvidia/nemotron-3-super-120b-a12b:free"],
)
def test_free_ids_are_accepted(model_id: str) -> None:
    assert is_free_model(model_id) is True
    assert assert_free_model(model_id) == model_id


@pytest.mark.parametrize(
    "model_id",
    [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.1-8b-instruct",  # paid variant, no :free suffix
        "openrouter/auto",
        "gpt-4o",
    ],
)
def test_paid_ids_are_rejected(model_id: str) -> None:
    assert is_free_model(model_id) is False
    with pytest.raises(NonFreeModelError) as excinfo:
        assert_free_model(model_id)
    assert model_id in str(excinfo.value)


def test_empty_id_is_rejected() -> None:
    with pytest.raises(NonFreeModelError):
        assert_free_model("   ")


def test_guard_error_names_the_free_rule() -> None:
    with pytest.raises(NonFreeModelError) as excinfo:
        assert_free_model("openai/gpt-4o")
    message = str(excinfo.value)
    assert ":free" in message
    assert "openrouter/free" in message


def test_candidates_filter_out_paid_models() -> None:
    models = ["openrouter/free", "openai/gpt-4o", "qwen/qwen3.8-27b:free", "", "  "]
    assert free_model_candidates(models) == ["openrouter/free", "qwen/qwen3.8-27b:free"]


def test_candidates_keep_order() -> None:
    models = ["qwen/qwen3.8-27b:free", "openrouter/free"]
    assert free_model_candidates(models) == models
