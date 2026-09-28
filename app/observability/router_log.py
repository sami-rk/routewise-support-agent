"""Router decision log.

Every routing decision is written to `router_decisions` with its full
probability distribution. This is what the routing eval reads, what the metrics
count, and what the trace view shows when a turn is questioned: which intent was
chosen, how confident the model was, and what the alternatives were.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..models.router import RouterDecision

logger = logging.getLogger(__name__)


def log_router_decision(thread_id: str, message: str, decision: RouterDecision) -> None:
    """Record one routing decision. Never raises."""
    try:
        from ..db.session import execute

        answers = decision.answers_for_log()
        probs = {
            "intent": decision.intent_probs,
            "needs_human": decision.needs_human_conf,
            "wants_human": decision.wants_human_conf,
            "sensitive": decision.sensitive_conf,
            "injection": decision.injection_conf,
            "frustrated": decision.frustrated_conf,
            "urgency": decision.urgency,
        }
        execute(
            "INSERT INTO router_decisions (thread_id, message, answers_json, probs_json) "
            "VALUES (?, ?, ?, ?)",
            (
                thread_id or "unknown",
                (message or "")[:2000],
                json.dumps(answers, ensure_ascii=False),
                json.dumps(probs, ensure_ascii=False),
            ),
        )
    except Exception:  # noqa: BLE001
        logger.debug("could not log a router decision", exc_info=True)


def recent_decisions(thread_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """The recent decisions for a thread, newest first, for the trace view."""
    from ..db.session import query_all

    rows = query_all(
        "SELECT * FROM router_decisions WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
        (thread_id, limit),
    )
    decisions: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("answers_json", "probs_json"):
            try:
                item[key.removesuffix("_json")] = json.loads(item.pop(key))
            except (ValueError, TypeError):
                item[key.removesuffix("_json")] = {}
        decisions.append(item)
    return decisions
