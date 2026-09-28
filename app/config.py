"""Application settings, loaded from the environment or a .env file.

Every LLM model id is validated here as well as in ``app.models.llm``: a non-free
id has to stop the process at import time, not on the first request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The OpenRouter free-model router, and any model whose id ends with ":free".
FREE_MODEL_IDS = frozenset({"openrouter/free"})

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_free_model(model_id: str) -> bool:
    """True when `model_id` is an OpenRouter free-tier model."""
    model_id = model_id.strip()
    return model_id in FREE_MODEL_IDS or model_id.endswith(":free")


class Settings(BaseSettings):
    """Configuration for the whole project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM layer: OpenRouter, free tier only -------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_models: list[str] = Field(default_factory=lambda: ["openrouter/free"])
    llm_temperature: float = 0.2
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 2

    # --- Router ------------------------------------------------------------
    router_backend: Literal["laya", "fake"] = "laya"
    router_checkpoint: str = "convaiinnovations/laya"
    router_device: Literal["auto", "cpu", "cuda"] = "auto"
    router_min_conf: float = 0.55

    # --- Refund policy -----------------------------------------------------
    refund_window_days: int = 14
    refund_auto_limit: float = 20.0

    # --- Knowledge base ----------------------------------------------------
    kb_dir: Path = Path("./data/knowledge_base")
    kb_top_k: int = 4
    kb_min_score: float = 0.35
    history_max_messages: int = 12

    # --- Storage -----------------------------------------------------------
    db_path: Path = Path("./data/support.db")
    checkpoint_db_path: Path = Path("./data/checkpoints.db")
    admin_api_key: str = "change-me"

    # --- Frontend ----------------------------------------------------------
    api_base_url: str = "http://localhost:8000"

    # --- Optional tracing --------------------------------------------------
    langfuse_enabled: bool = False

    @field_validator("llm_models", mode="before")
    @classmethod
    def _split_models(cls, value: object) -> object:
        """Accept `a,b,c` from the environment as well as a real list."""
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):  # JSON-ish, for programmatic overrides
                import json

                return json.loads(raw)
            return [part.strip() for part in raw.split(",") if part.strip()]
        return value

    @field_validator("llm_models")
    @classmethod
    def _reject_paid_models(cls, value: list[str]) -> list[str]:
        """Refuse any model that is not free-tier, at startup."""
        if not value:
            raise ValueError("LLM_MODELS is empty; set at least one free OpenRouter model")
        paid = [m for m in value if not is_free_model(m)]
        if paid:
            raise ValueError(
                f"LLM_MODELS contains non-free models {paid}. This project may only use "
                f"OpenRouter free-tier models: ids ending in ':free' or 'openrouter/free'."
            )
        return value

    @field_validator("kb_dir", "db_path", "checkpoint_db_path", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        """Resolve relative paths against the project root, not the cwd."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _usable_budget(self) -> Settings:
        if self.llm_timeout_s <= 0:
            raise ValueError("LLM_TIMEOUT_S must be positive")
        if self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES cannot be negative")
        if not 0.0 < self.router_min_conf <= 1.0:
            raise ValueError("ROUTER_MIN_CONF must be in (0, 1]")
        if self.kb_top_k < 1:
            raise ValueError("KB_TOP_K must be at least 1")
        return self

    @property
    def has_llm_key(self) -> bool:
        return bool(self.openrouter_api_key.strip())

    @property
    def kb_index_dir(self) -> Path:
        """Where the persisted FAISS index lives, next to the documents."""
        return self.kb_dir.parent / "kb_index"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Used by tests that override the environment."""
    get_settings.cache_clear()
