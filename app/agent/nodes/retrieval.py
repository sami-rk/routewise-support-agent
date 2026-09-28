"""Retrieval: search the knowledge base and fill `retrieved_context`.

When nothing clears the similarity floor the agent is told so, and says it does
not know rather than answering from an unrelated passage. That is the difference
between a grounded answer and a confident invention.
"""

from __future__ import annotations

from typing import Any

from ...rag.retriever import get_retriever
from ..state import SupportState

# What the agent is given when retrieval found nothing. Distinct from an empty
# string so the prompt can say "nothing was found" rather than leaving a gap.
NOTHING_FOUND = "(nothing relevant was found in the knowledge base)"


def retrieve_kb(state: SupportState) -> dict[str, Any]:
    """Search the knowledge base for this turn's question.

    The retriever carries its own `top_k` and `min_score`, read from settings
    when it was loaded. This node does not override them: two sources of truth
    for one threshold is how a tuned retriever silently stops being used.

    Returns:
        `retrieved` (the passages) and `retrieved_context` (the flattened text
        with citations). `retrieved_context` is `NOTHING_FOUND` when the search
        came back empty, so the agent does not try to answer from nothing.
    """
    query = (state.get("retrieval_query") or state.get("user_input") or "").strip()
    if not query:
        return {"retrieved": [], "retrieved_context": NOTHING_FOUND}

    try:
        retriever = get_retriever()
        # The embedder is lazy, so the model is loaded here, on the first query,
        # and not when the retriever is built. Both halves have to be covered:
        # a missing index raises at `get_retriever`, a missing
        # sentence-transformers raises here.
        passages = retriever.search(query)
    except Exception as exc:  # noqa: BLE001
        # A knowledge base that cannot be read is a degraded system, not a
        # broken one: the agent should say it does not know rather than fail
        # the turn. FileNotFoundError, ImportError and ValueError from a
        # half-written index all land here.
        return {
            "retrieved": [],
            "retrieved_context": f"{NOTHING_FOUND} (retrieval unavailable: {type(exc).__name__})",
            "retrieval_error": f"{type(exc).__name__}: {exc}",
        }

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
