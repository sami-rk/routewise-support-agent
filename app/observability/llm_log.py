"""One row per LLM attempt, in the `llm_calls` table.

The per-node `traces` table answers "what happened in the graph"; this answers
"how many requests did we spend, on which model, and how many were rate
limited", which is what the free tier makes worth measuring.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def record_call(
    *,
    role: str,
    model: str,
    attempt: int,
    status: str,
    error: str | None = None,
    duration_ms: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    run_id: str | None = None,
    thread_id: str | None = None,
) -> None:
    """Write one attempt, never raising.

    Observability must not be able to break a conversation, so a failure to log
    is logged and swallowed.
    """
    try:
        from ..db.session import execute

        execute(
            "INSERT INTO llm_calls (run_id, thread_id, role, model, attempt, status, error, "
            "duration_ms, tokens_in, tokens_out) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                thread_id,
                role,
                model,
                attempt,
                status,
                (error or None) and error[:500],
                round(duration_ms, 2),
                tokens_in,
                tokens_out,
            ),
        )
    except Exception:  # noqa: BLE001
        logger.debug("could not record llm call %s/%s", role, model, exc_info=True)
