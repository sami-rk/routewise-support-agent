"""Every LLM call in this project goes through here.

The rules this module enforces, in order of how much they matter:

* **Free tier only.** `get_llm` refuses any model id that is not `openrouter/free`
  and does not end in `:free`, so a paid model cannot be configured even by
  accident. `app.config` refuses the same thing at startup, which is what stops
  the server booting with a paid id in its environment.
* **Few requests.** Free models are limited to about 20 requests a minute and 50
  a day without purchased credits, so a turn is one call, tools loop at most
  `LLM_MAX_STEPS` times, and `update_memory` runs at the end of a thread.
* **Never a stack trace.** Rate limits and upstream failures are retried with
  backoff, then answered on the next configured model, and finally turned into a
  friendly message.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..config import Settings, get_settings
from .free_guard import assert_free_model, free_model_candidates

# Which model to use when one is good enough for every task. `openrouter/free` is
# OpenRouter's own free-model router; it picks a free model that supports the
# features asked for, which matters because the billing agent needs tool calling.
ROLES = ("faq", "billing", "technical", "escalate", "clarify", "memory", "ticket")

# Errors worth retrying on another model: rate limits and transient upstream
# failures. A 400 means the request is wrong, so retrying it changes nothing.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class LLMUnavailableError(RuntimeError):
    """No configured free model could answer. The message is customer-safe."""


BUSY_MESSAGE = (
    "Our assistant is getting a lot of requests right now, so I could not finish "
    "that just yet. Please try again in a minute or two, and sorry about the wait."
)


@dataclass
class LLMCall:
    """One attempt, for the `llm_calls` table and the metrics."""

    role: str
    model: str
    attempt: int
    status: str
    error: str | None = None
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    run_id: str | None = None
    thread_id: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model,
            "attempt": self.attempt,
            "status": self.status,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
        }


def models_for_role(settings: Settings | None = None) -> list[str]:
    """The free models to try, in order, for any role.

    Tool calling is only reliable on some free models, so a role that binds tools
    prefers `openrouter/free`, which routes to a model that supports them.
    """
    settings = settings or get_settings()
    candidates = free_model_candidates(settings.llm_models)
    if not candidates:
        raise LLMUnavailableError(
            "LLM_MODELS contains no free OpenRouter model. Set it to openrouter/free "
            "or to ids ending in ':free'."
        )
    return candidates


def build_chat_model(
    model: str,
    settings: Settings | None = None,
    *,
    tools: list[Any] | None = None,
) -> Any:
    """Build a `ChatOpenAI` pointed at OpenRouter, for one model.

    Args:
        model: a free OpenRouter model id. Anything else is refused.
        settings: for the key, base url, temperature and timeout.
        tools: tool schemas to bind.

    Returns:
        A ready-to-invoke chat model.

    Raises:
        NonFreeModelError: if `model` is not a free-tier model.
        LLMUnavailableError: if no OpenRouter key is configured.
    """
    settings = settings or get_settings()
    model = assert_free_model(model)

    if not settings.has_llm_key:
        raise LLMUnavailableError(
            "OPENROUTER_API_KEY is not set. Add it to .env to make live LLM calls; "
            "the test suite runs without one."
        )

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        # The free router and some free models reject an explicit max_tokens, so
        # leave it unset and let the provider decide.
    )
    if tools:
        llm = llm.bind_tools(tools)
    return llm


def get_llm(
    role: str = "faq",
    settings: Settings | None = None,
    *,
    tools: list[Any] | None = None,
) -> Any:
    """The chat model for a role, on the first configured free model.

    Args:
        role: which job it does, for logging and tool choice.
        settings: settings to use.
        tools: tool schemas to bind.

    Returns:
        A ready-to-invoke chat model.

    Raises:
        NonFreeModelError: if the configured models are not all free-tier.
        LLMUnavailableError: if no key is configured.
    """
    model = models_for_role(settings)[0]
    return build_chat_model(model, settings, tools=tools)


def get_llm_chain(
    role: str = "faq",
    settings: Settings | None = None,
    *,
    tools: list[Any] | None = None,
) -> list[tuple[str, Any]]:
    """Every candidate model for a role, in fallback order."""
    settings = settings or get_settings()
    chain: list[tuple[str, Any]] = []
    for model in models_for_role(settings):
        try:
            chain.append((model, build_chat_model(model, settings, tools=tools)))
        except Exception:  # a bad entry should not hide the rest
            continue
    if not chain:
        raise LLMUnavailableError("None of the configured free models could be built.")
    return chain


def status_of(error: BaseException) -> int | None:
    """The HTTP status of a provider error, however it was surfaced.

    Clients differ: some raise with a `status_code` attribute, some put the code
    at the front of the message. Both are checked, because deciding whether to
    retry depends on it.
    """
    status = getattr(error, "status_code", None) or getattr(error, "http_status", None)
    if isinstance(status, int):
        return status
    match = re.search(r"\b([45]\d{2})\b", str(error))
    return int(match.group(1)) if match else None


def is_rate_limited(error: BaseException) -> bool:
    """True when the failure is a rate limit we should back off from."""
    text = str(error).lower()
    if "rate limit" in text or "too many requests" in text:
        return True
    return status_of(error) == 429


def is_retryable(error: BaseException) -> bool:
    """True when another attempt, or another model, could plausibly work."""
    if is_rate_limited(error):
        return True
    text = str(error).lower()
    if any(marker in text for marker in ("timeout", "timed out", "temporarily", "overloaded")):
        return True
    return status_of(error) in RETRYABLE_STATUS


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff for attempt N, capped so a turn cannot hang.

    Args:
        attempt: 1 for the first retry.
        base: seconds for the first retry.
        cap: ceiling on any single wait.

    Returns:
        Seconds to sleep. 1, 2, 4, 8 ... up to `cap`.
    """
    return min(cap, base * (2 ** max(0, attempt - 1)))


def sleep_for(seconds: float) -> None:
    """Indirection so tests can patch the wait out."""
    if seconds > 0:
        time.sleep(seconds)
