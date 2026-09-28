"""Long-term customer memory.

`customer_memory` holds one summary per customer, at most 150 words, merged at
the end of a thread rather than written every turn — on the free tier an extra
call per turn is not affordable, and most turns add nothing durable.

The summary is injected into the agent's system prompt by `load_context`, so a
customer does not have to repeat their plan or their platform every session.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from ..db.session import execute, query_one

UTC = timezone.utc

MAX_WORDS = 150
# The model is told to reply with exactly this when there is nothing to keep.
NOTHING = "NOTHING"


def get_memory(customer_id: str) -> str:
    """The stored summary for a customer, or an empty string."""
    if not customer_id:
        return ""
    row = query_one(
        "SELECT summary FROM customer_memory WHERE customer_id = ?", (customer_id,)
    )
    return (row["summary"] if row else "") or ""


def set_memory(customer_id: str, summary: str, *, now: datetime | None = None) -> str:
    """Replace a customer's summary, trimmed to the word limit.

    Returns:
        The summary as stored.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    trimmed = trim_summary(summary)
    execute(
        "INSERT INTO customer_memory (customer_id, summary, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(customer_id) DO UPDATE SET summary = excluded.summary, "
        "updated_at = excluded.updated_at",
        (customer_id, trimmed, stamp),
    )
    return trimmed


def clear_memory(customer_id: str) -> bool:
    """Forget a customer. Returns whether there was anything to forget."""
    if not customer_id:
        return False
    row = query_one("SELECT customer_id FROM customer_memory WHERE customer_id = ?", (customer_id,))
    execute("DELETE FROM customer_memory WHERE customer_id = ?", (customer_id,))
    return row is not None


def trim_summary(summary: str) -> str:
    """Collapse whitespace and cut to `MAX_WORDS` on a sentence boundary.

    A summary cut mid-sentence is worse than one cut short, so the last
    complete sentence wins when the limit lands inside one.
    """
    text = re.sub(r"\s+", " ", (summary or "").strip())
    if not text:
        return ""
    words = text.split()
    if len(words) <= MAX_WORDS:
        return text

    shortened = " ".join(words[:MAX_WORDS])
    # Prefer ending on a sentence if one is close enough to the cut.
    for terminator in (". ", "? ", "! "):
        cut = shortened.rfind(terminator)
        if cut > len(shortened) * 0.6:
            return shortened[: cut + 1].strip()
    return shortened.rstrip(",;:") + "."


def is_worth_storing(summary: str) -> bool:
    """False when the model said there was nothing durable to keep."""
    text = (summary or "").strip().upper()
    if not text or text == NOTHING:
        return False
    return len(text.split()) >= 5  # a stub word or two is not a memory


def customer_facts(customer_id: str) -> dict[str, Any]:
    """The facts the prompt shows: name, plan, devices, and the memory summary."""
    from ..tools.customers import get_customer, get_subscription

    profile = get_customer(customer_id) if customer_id else None
    subscription = get_subscription(customer_id) if customer_id else None
    devices = None
    if subscription:
        allowed = subscription["devices_allowed"]
        devices = (
            f"{subscription['devices_used']} of unlimited"
            if allowed == 0
            else f"{subscription['devices_used']} of {allowed}"
        )
    return {
        "name": profile["name"] if profile else None,
        "email": profile["email"] if profile else None,
        "plan": subscription["plan"] if subscription else None,
        "devices": devices,
        "memory": get_memory(customer_id) or None,
    }
