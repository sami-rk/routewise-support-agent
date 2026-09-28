"""Tests for application settings, including free-model enforcement at startup."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings, is_free_model


def make_settings(**overrides: object) -> Settings:
    """Build settings from the environment without touching a real .env."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_defaults_use_the_free_router() -> None:
    settings = make_settings()
    assert settings.llm_models == ["openrouter/free"]
    assert settings.llm_temperature == 0.2
    assert settings.refund_window_days == 14
    assert settings.refund_auto_limit == 20.0
    assert settings.router_min_conf == 0.55


def test_model_list_is_split_from_the_environment() -> None:
    settings = make_settings(llm_models="openrouter/free,qwen/qwen3.8-27b:free")
    assert settings.llm_models == ["openrouter/free", "qwen/qwen3.8-27b:free"]


def test_paid_model_in_the_list_stops_startup() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_settings(llm_models="openrouter/free,openai/gpt-4o")
    message = str(excinfo.value)
    assert "openai/gpt-4o" in message
    assert ":free" in message


def test_empty_model_list_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_models="  ,  ")


def test_json_style_model_list_is_accepted() -> None:
    settings = make_settings(llm_models='["openrouter/free"]')
    assert settings.llm_models == ["openrouter/free"]


def test_a_comma_separated_env_var_is_split(monkeypatch) -> None:
    # The environment is what a container and CI set. pydantic-settings tries to
    # JSON-parse a list field before the validator runs, so this only works
    # because of the `NoDecode` annotation on the field; without it this raises.
    from app.config import get_settings, reset_settings_cache

    monkeypatch.setenv("LLM_MODELS", "openrouter/free,qwen/qwen3.8-27b:free")
    reset_settings_cache()
    try:
        assert get_settings().llm_models == ["openrouter/free", "qwen/qwen3.8-27b:free"]
    finally:
        reset_settings_cache()


def test_relative_paths_resolve_against_the_project_root() -> None:
    settings = make_settings(db_path=Path("./data/support.db"))
    assert settings.db_path == PROJECT_ROOT / "data" / "support.db"
    assert settings.kb_dir == PROJECT_ROOT / "data" / "knowledge_base"


def test_kb_index_sits_next_to_the_documents() -> None:
    settings = make_settings(kb_dir=Path("./data/knowledge_base"))
    assert settings.kb_index_dir == PROJECT_ROOT / "data" / "kb_index"


def test_absolute_paths_are_left_alone() -> None:
    settings = make_settings(db_path=Path("/tmp/custom.db"))
    assert settings.db_path == Path("/tmp/custom.db")


def test_missing_llm_key_is_not_an_error() -> None:
    settings = make_settings(openrouter_api_key="")
    assert settings.has_llm_key is False
    assert make_settings(openrouter_api_key="sk-or-abc").has_llm_key is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"router_min_conf": 0.0},
        {"router_min_conf": 1.5},
        {"llm_timeout_s": 0},
        {"llm_max_retries": -1},
        {"kb_top_k": 0},
        {"router_backend": "gpt"},
    ],
)
def test_out_of_range_settings_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_free_model_helper_matches_the_guard() -> None:
    assert is_free_model("openrouter/free")
    assert is_free_model("anything/at-all:free")
    assert not is_free_model("vendor/model")
