"""The memory node: fold a finished thread into the customer's long-term summary.

This runs at the end of a turn, but it only calls the model when the thread has
something durable to record and only every `MEMORY_EVERY` turns, because on the
free tier an extra request per turn is not affordable and most turns add nothing
worth keeping. A customer who says "thanks" does not need an LLM call.
"""

from __future__ import annotations

import logging
from typing import Any

from ...memory.long_term import is_worth_storing, set_memory, trim_summary
from ...models.llm import LLMUnavailableError
from ...models.llm_runner import invoke_with_fallback
from .. import prompts
from ..state import SupportState

logger = logging.getLogger(__name__)

# Only summarise this often. A refund conversation is worth remembering sooner
# than a run of smalltalk.
MEMORY_EVERY = 3
# Below this many turns there is nothing to summarise.
MIN_TURNS = 2


def should_update_memory(state: SupportState, every: int = MEMORY_EVERY) -> bool:
    """Whether this turn should produce a memory update.

    Skipped for turns that were not worth remembering: an injection attempt, a
    plain greeting, or a handoff, none of which say anything durable about the
    customer.
    """
    if state.get("unsafe"):
        return False
    if not state.get("customer_id"):
        return False

    turns = len([m for m in (state.get("messages") or []) if getattr(m, "type", "") in ("human", "ai")])
    if turns < MIN_TURNS:
        return False
    if every > 0 and turns % every != 0:
        return False

    intent = state.get("intent")
    return intent not in ("smalltalk", None)


def update_memory(state: SupportState, every: int = MEMORY_EVERY) -> dict[str, Any]:
    """Summarise durable facts and store them against the customer."""
    from langchain_core.messages import HumanMessage, SystemMessage

    customer_id = state.get("customer_id")
    if not should_update_memory(state, every):
        return {"memory_updated": False}

    from ...memory.long_term import get_memory

    existing = state.get("customer_memory") or get_memory(customer_id or "")
    turns = _turns_for_memory(state)

    try:
        response = invoke_with_fallback(
            [
                SystemMessage(prompts.memory_prompt(existing, turns)),
                HumanMessage(state.get("user_input") or ""),
            ],
            role="memory",
            run_id=state.get("run_id"),
            thread_id=state.get("thread_id"),
        )
    except LLMUnavailableError as exc:
        # Losing a memory update is not worth failing a conversation over.
        logger.info("memory update skipped: %s", exc)
        return {"memory_updated": False}

    summary = _text(response)
    if not is_worth_storing(summary):
        return {"memory_updated": False}

    stored = set_memory(customer_id or "", trim_summary(summary))
    return {"memory_updated": True, "customer_memory": stored}


def _text(response: Any) -> str:
    from .shared import message_text

    return message_text(response)


def _turns_for_memory(state: SupportState, keep: int = 6) -> list[tuple[str, str]]:
    """The recent conversation, as `(role, text)` pairs."""
    from langchain_core.messages import AIMessage, HumanMessage

    turns: list[tuple[str, str]] = []
    for message in (state.get("messages") or []):
        if isinstance(message, HumanMessage):
            turns.append(("Customer", str(message.content)))
        elif isinstance(message, AIMessage):
            turns.append(("Agent", str(message.content)))
    return turns[-keep:]
