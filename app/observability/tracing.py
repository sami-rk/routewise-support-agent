"""Per-node tracing.

Every node is wrapped so each run records what it was given, what it returned,
how long it took, which model it used, and whether it failed. The record goes to
two places: the `traces` table, and `state["trace"]`, which the API returns for a
thread so the UI can draw a timeline.

Summaries are truncated, because a trace is for reading, not for storing whole
conversations.
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from typing import Any, Callable, TypeVar

from langgraph.errors import GraphInterrupt

from ..db.session import execute
from .summaries import summarise_output, summarise_input

logger = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 600

F = TypeVar("F", bound=Callable[..., Any])


def new_run_id() -> str:
    """An id for one turn, shared by every node in that turn."""
    return uuid.uuid4().hex[:16]


def record_trace(
    *,
    run_id: str,
    thread_id: str,
    node: str,
    started_at: str,
    duration_ms: float,
    input_summary: str = "",
    output_summary: str = "",
    model: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    error: str | None = None,
) -> None:
    """Write one node trace. Never raises: tracing must not break a turn."""
    try:
        execute(
            "INSERT INTO traces (run_id, thread_id, node, started_at, duration_ms, "
            "input_summary, output_summary, model, tokens_in, tokens_out, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                thread_id,
                node,
                started_at,
                round(duration_ms, 2),
                _clip(input_summary),
                _clip(output_summary),
                model,
                tokens_in,
                tokens_out,
                _clip(error) if error else None,
            ),
        )
    except Exception:  # noqa: BLE001
        logger.debug("could not record trace for node %s", node, exc_info=True)


def _clip(text: str | None) -> str:
    if not text:
        return ""
    text = str(text)
    return text if len(text) <= MAX_SUMMARY_CHARS else text[:MAX_SUMMARY_CHARS] + "..."


def trace_node(node_name: str, *, model: str | None = None) -> Callable[[F], F]:
    """Wrap a graph node so each call is traced.

    The wrapper adds `run_id` to the state on the way in if it is missing, and
    appends the trace entry to `state["trace"]` on the way out, so callers do not
    have to remember to. A node that raises is still traced, with the error, and
    then the error is re-raised.

    Args:
        node_name: the name to record.
        model: the model the node uses, when known.

    Returns:
        A decorator.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            run_id = state.get("run_id") or new_run_id()
            thread_id = state.get("thread_id") or "unknown"
            started_at = _now()
            started = time.perf_counter()

            try:
                result = func({**state, "run_id": run_id}, *args, **kwargs) or {}
            except GraphInterrupt:
                # `interrupt()` pauses the graph by raising. That is the design
                # working, not a failure, so it must not count as an error in
                # /metrics.
                record_trace(
                    run_id=run_id,
                    thread_id=thread_id,
                    node=node_name,
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    input_summary=summarise_input(state),
                    output_summary="interrupted",
                    model=model,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                record_trace(
                    run_id=run_id,
                    thread_id=thread_id,
                    node=node_name,
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    input_summary=summarise_input(state),
                    output_summary="",
                    model=model,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

            duration_ms = (time.perf_counter() - started) * 1000
            entry = {
                "node": node_name,
                "run_id": run_id,
                "thread_id": thread_id,
                "started_at": started_at,
                "duration_ms": round(duration_ms, 2),
                "input": summarise_input(state),
                "output": summarise_output(result),
                "model": model,
            }
            record_trace(
                run_id=run_id,
                thread_id=thread_id,
                node=node_name,
                started_at=started_at,
                duration_ms=duration_ms,
                input_summary=entry["input"],
                output_summary=entry["output"],
                model=model,
            )
            return {**result, "run_id": run_id, "trace": [*(state.get("trace") or []), entry]}

        return wrapper  # type: ignore[return-value]

    return decorate


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_thread_trace(thread_id: str) -> list[dict[str, Any]]:
    """Every trace row for a thread, in order, for the timeline view."""
    from ..db.session import query_all

    rows = query_all(
        "SELECT * FROM traces WHERE thread_id = ? ORDER BY id ASC", (thread_id,)
    )
    return [dict(row) for row in rows]
