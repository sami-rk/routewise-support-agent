"""Calling a free model without ever surfacing a stack trace.

`invoke_with_fallback` is the only path a node uses to talk to a model. It walks
the configured free models in order, retries each one with exponential backoff
while the failure looks transient, records every attempt in `llm_calls`, and
either returns an answer or raises `LLMUnavailableError` carrying the friendly
message to show the customer.

The wait is capped and the whole thing is bounded by `max_steps`, because a turn
on a rate-limited account has to fail in seconds, not minutes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..config import Settings, get_settings
from ..observability.llm_log import record_call
from .llm import (
    BUSY_MESSAGE,
    LLMUnavailableError,
    backoff_delay,
    get_llm_chain,
    is_rate_limited,
    is_retryable,
    sleep_for,
)

logger = logging.getLogger(__name__)

# How many models to try before giving up on the turn.
MAX_MODELS = 3
# How many extra attempts on the same model.
MAX_ATTEMPTS = 2


def invoke_with_fallback(
    prompt: Any,
    role: str = "faq",
    settings: Settings | None = None,
    *,
    tools: list[Any] | None = None,
    max_steps: int = MAX_MODELS,
    record: bool = True,
    run_id: str | None = None,
    thread_id: str | None = None,
    **invoke_kwargs: Any,
) -> Any:
    """Call the configured free models until one answers.

    Args:
        prompt: messages, or anything the chat model's `invoke` accepts.
        role: which node is calling, for logging and metrics.
        settings: settings to use.
        tools: tool schemas to bind.
        max_steps: how many models to try.
        record: write an `llm_calls` row per attempt. Off in tests.
        run_id: the graph run, so a call can be tied to a turn.
        thread_id: the conversation, for the same reason.
        invoke_kwargs: passed through to `invoke`.

    Returns:
        The model's response message.

    Raises:
        LLMUnavailableError: when every model failed. `str(error)` is the
            customer-facing message, never a traceback.
    """
    settings = settings or get_settings()
    models = get_llm_chain(role, settings, tools=tools)[:max_steps]
    if not models:
        raise LLMUnavailableError(BUSY_MESSAGE)

    last_error: BaseException | None = None

    for index, (model_name, llm) in enumerate(models):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = _now()
            try:
                response = llm.invoke(prompt, **invoke_kwargs)
            except Exception as exc:  # noqa: BLE001 - any provider error is handled the same way
                last_error = exc
                duration = (_now() - started) * 1000
                limited = is_rate_limited(exc)
                if record:
                    record_call(
                        role=role,
                        model=model_name,
                        attempt=attempt,
                        status="rate_limited" if limited else "error",
                        error=str(exc)[:500],
                        duration_ms=duration,
                        run_id=run_id,
                        thread_id=thread_id,
                    )
                if not is_retryable(exc):
                    # A 400 will not become a 200 on the same model, but the next
                    # model may accept the request, so stop retrying this one only.
                    logger.info("llm %s failed unretryably: %s", model_name, exc)
                    break
                if attempt < MAX_ATTEMPTS:
                    sleep_for(backoff_delay(attempt))
                    continue
                break
            else:
                if record:
                    record_call(
                        role=role,
                        model=model_name,
                        attempt=attempt,
                        status="ok",
                        duration_ms=(_now() - started) * 1000,
                        tokens_in=_tokens(response, "input"),
                        tokens_out=_tokens(response, "output"),
                        run_id=run_id,
                        thread_id=thread_id,
                    )
                return response
        # On to the next model, if there is one.
        if index < len(models) - 1:
            logger.info("falling back from %s after a failed call", model_name)

    logger.warning("every configured free model failed for role %s: %s", role, last_error)
    raise LLMUnavailableError(BUSY_MESSAGE) from last_error


def _now() -> float:
    import time

    return time.perf_counter()


def _tokens(response: Any, key: str) -> int:
    """Pull a token count off a response, tolerating providers that omit it."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get(key) or 0)
    metadata = getattr(response, "response_metadata", None) or {}
    tokens = metadata.get("token_usage") or metadata.get("usage") or {}
    if isinstance(tokens, dict):
        return int(tokens.get(key) or tokens.get(key.replace("input", "prompt")) or 0)
    return 0
