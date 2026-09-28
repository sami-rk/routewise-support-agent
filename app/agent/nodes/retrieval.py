"""Retrieval: search the knowledge base and fill `retrieved_context`.

When nothing clears the similarity floor the agent is told so, and says it does
not know rather than answering from an unrelated passage. That is the difference
between a grounded answer and a confident invention.
"""

from __future__ import annotations

from typing import Any

from ...config import Settings, get_settings
from ...rag.retriever import get_retriever
from ..state import SupportState

# What the agent is given when retrieval found nothing. Distinct from an empty
# string so the prompt can say "nothing was found" rather than leaving a gap.
NOTHING_FOUND = "(nothing relevant was found in the knowledge base)"


def retrieve_kb(state: SupportState, settings: Settings | None = None) -> dict[str, Any]:
    """Search the knowledge base for this turn's question.

    Returns:
        `retrieved` (the passages) and `retrieved_context` (the flattened text
        with citations). `retrieved_context` is `NOTHING_FOUND` when the search
        came back empty, so the agent does not try to answer from nothing.
    """
    settings = settings or get_settings()
    query = (state.get("retrieval_query") or state.get("user_input") or "").strip()
    if not query:
        return {"retrieved": [], "retrieved_context": NOTHING_FOUND}

    try:
        retriever = get_retriever()
    except FileNotFoundError as exc:
        # No index built yet. Tell the agent, rather than failing the turn.
        return {
            "retrieved": [],
            "retrieved_context": f"{NOTHING_FOUND} (the knowledge base is not built: {exc})",
        }

    passages = retriever.search(query, top_k=settings.kb_top_k, min_score=settings.kb_min_score)
    if not passages:
        return {"retrieved": [], "retrieved_context": NOTHING_FOUND}

    context = "\n\n".join(f"{p['citation']}\n{p['text']}" for p in passages)
    return {"retrieved": passages, "retrieved_context": context}


def citations_for(state: SupportState) -> list[str]:
    """The citation strings for the passages this turn retrieved."""
    return [str(passage.get("citation", "")) for passage in state.get("retrieved") or []]


def has_context(state: SupportState) -> bool:
    """Whether anything usable was retrieved."""
    return bool(state.get("retrieved"))
