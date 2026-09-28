"""Aggregated metrics for `GET /metrics`.

Everything here is a real number read from the tables — there is no in-process
counter that drifts from reality. Latency comes from the `traces` table, routing
from `router_decisions`, and the LLM cost from `llm_calls`.
"""

from __future__ import annotations

from typing import Any

from ..db.session import query_all, query_one


def _scalar(sql: str, params: tuple = ()) -> Any:
    row = query_one(sql, params)
    if row is None:
        return None
    return row[0]


def percentile(values: list[float], fraction: float) -> float:
    """The `fraction` percentile, nearest-rank.

    With a handful of samples an interpolated percentile invents precision that is
    not there, so this picks an actual observed value.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 2)


def collect() -> dict[str, Any]:
    """Every metric, in one dict."""
    return {
        "conversations": conversation_metrics(),
        "routing": routing_metrics(),
        "outcomes": outcome_metrics(),
        "llm": llm_metrics(),
        "refunds": refund_metrics(),
    }


def conversation_metrics() -> dict[str, Any]:
    """Volume and latency, from the traces table."""
    total = int(_scalar("SELECT COUNT(DISTINCT thread_id) FROM traces") or 0)
    # A turn is the run: one run_id per customer message.
    turns = int(_scalar("SELECT COUNT(DISTINCT run_id) FROM traces") or 0)

    # One duration per turn: the slowest node in each run stands for the turn.
    per_turn = query_all(
        "SELECT run_id, SUM(duration_ms) AS total FROM traces GROUP BY run_id"
    )
    latencies = [float(row["total"]) for row in per_turn]
    node_latencies = [float(row["duration_ms"]) for row in query_all("SELECT duration_ms FROM traces")]

    return {
        "threads": total,
        "turns": turns,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "avg_node_latency_ms": (
            round(sum(node_latencies) / len(node_latencies), 2) if node_latencies else 0.0
        ),
        "p95_node_latency_ms": percentile(node_latencies, 0.95),
        "errors": int(_scalar("SELECT COUNT(*) FROM traces WHERE error IS NOT NULL") or 0),
    }


def routing_metrics() -> dict[str, Any]:
    """Intent distribution and router confidence, from `router_decisions`."""
    rows = query_all("SELECT answers_json FROM router_decisions")
    intents: dict[str, int] = {}
    confidences: list[float] = []
    backends: dict[str, int] = {}
    latencies: list[float] = []

    for row in rows:
        try:
            import json

            answers = json.loads(row["answers_json"])
        except (ValueError, TypeError):
            continue
        intent = answers.get("intent")
        if intent:
            intents[intent] = intents.get(intent, 0) + 1
        if isinstance(answers.get("intent_conf"), (int, float)):
            confidences.append(float(answers["intent_conf"]))
        backend = answers.get("backend") or "unknown"
        backends[backend] = backends.get(backend, 0) + 1
        if isinstance(answers.get("latency_ms"), (int, float)):
            latencies.append(float(answers["latency_ms"]))

    total = sum(intents.values())
    return {
        "decisions": len(rows),
        "intent_distribution": dict(sorted(intents.items(), key=lambda kv: -kv[1])),
        "intent_share": {k: round(v / total, 4) for k, v in intents.items()} if total else {},
        "avg_intent_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        ),
        "low_confidence_share": (
            round(sum(1 for c in confidences if c < 0.55) / len(confidences), 4)
            if confidences
            else 0.0
        ),
        "by_backend": dict(sorted(backends.items(), key=lambda kv: -kv[1])),
        "router_latency": {
            "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95_ms": percentile(latencies, 0.95),
        },
    }


def outcome_metrics() -> dict[str, Any]:
    """Escalation and auto-resolution rates.

    A turn is resolved by the assistant when it ended without a ticket, and was
    escalated when it opened one.
    """
    escalations = int(_scalar("SELECT COUNT(*) FROM tickets") or 0)
    turns = int(_scalar("SELECT COUNT(DISTINCT run_id) FROM traces") or 0)
    # Runs that reached the escalate node, read from the trace node names.
    escalated_runs = int(
        _scalar("SELECT COUNT(DISTINCT run_id) FROM traces WHERE node = 'escalate'") or 0
    )
    return {
        "tickets": escalations,
        "escalated_runs": escalated_runs,
        "escalation_rate": round(escalated_runs / turns, 4) if turns else 0.0,
        "auto_resolution_rate": (
            round((turns - escalated_runs) / turns, 4) if turns else 0.0
        ),
    }


def llm_metrics() -> dict[str, Any]:
    """Requests per model, failures, fallbacks and rate-limit hits."""
    rows = query_all(
        "SELECT model, status, COUNT(*) AS n FROM llm_calls GROUP BY model, status"
    )
    per_model: dict[str, int] = {}
    per_model_errors: dict[str, int] = {}
    total = 0
    ok = 0
    rate_limited = 0
    errors = 0

    for row in rows:
        model = row["model"]
        count = int(row["n"])
        status = row["status"]
        total += count
        per_model[model] = per_model.get(model, 0) + count
        if status == "ok":
            ok += count
        elif status == "rate_limited":
            rate_limited += count
        else:
            errors += count
            per_model_errors[model] = per_model_errors.get(model, 0) + count

    # A fallback is any failed attempt on a model that was not the first one
    # tried, approximated by counting failed attempts, since a turn that failed
    # on one model and succeeded on the next had to make more than one request.
    successful_models = int(
        _scalar("SELECT COUNT(*) FROM llm_calls WHERE status = 'ok'") or 0
    )
    fallbacks = max(0, total - successful_models - errors - rate_limited)

    return {
        "total_requests": total,
        "successful": ok,
        "errors": errors,
        "rate_limited": rate_limited,
        "fallbacks": fallbacks,
        "error_rate": round((errors + rate_limited) / total, 4) if total else 0.0,
        "requests_per_model": dict(sorted(per_model.items(), key=lambda kv: -kv[1])),
        "errors_per_model": per_model_errors,
    }


def refund_metrics() -> dict[str, Any]:
    """Refunds issued, and how many were automatic rather than approved."""
    total = int(_scalar("SELECT COUNT(*) FROM refunds") or 0)
    automatic = int(
        _scalar("SELECT COUNT(*) FROM refunds WHERE approved_by = 'automatic'") or 0
    )
    amounts = query_all("SELECT amount FROM refunds")
    total_amount = round(sum(float(row["amount"]) for row in amounts), 2)
    decided = automatic + int(
        _scalar("SELECT COUNT(*) FROM refunds WHERE approved_by IS NOT NULL AND approved_by != 'automatic'")
        or 0
    )
    return {
        "refunds": total,
        "total_refunded": total_amount,
        "auto_approved": automatic,
        "staff_approved": decided - automatic,
        "approval_rate": round(automatic / decided, 4) if decided else 0.0,
        "pending": int(
            _scalar("SELECT COUNT(*) FROM pending_actions WHERE status = 'awaiting'") or 0
        ),
    }
