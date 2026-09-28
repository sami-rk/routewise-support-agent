"""Search the knowledge base and return passages with their citations.

Two things matter here for honesty of answers:

* a similarity floor, so a question the knowledge base cannot answer comes back
  with nothing rather than with the least-bad unrelated passage, and
* citations, so the agent can point at where an answer came from.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from .indexer import Passage, load_index


class Retriever:
    """A loaded FAISS index, searched with a local embedder."""

    def __init__(
        self,
        index: Any,
        passages: list[Passage],
        embedder: Any,
        *,
        top_k: int = 4,
        min_score: float = 0.35,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.passages = passages
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.metadata = metadata or {}

    @classmethod
    def load(cls, settings: Settings | None = None, embedder: Any = None) -> "Retriever":
        """Load the index that `scripts/build_kb.py` wrote.

        Raises:
            FileNotFoundError: if the index has not been built, with the command
                to build it.
        """
        settings = settings or get_settings()
        index, passages, metadata = load_index(settings.kb_index_dir / "index.faiss")
        if embedder is None:
            from .ingest import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        return cls(
            index,
            passages,
            embedder,
            top_k=settings.kb_top_k,
            min_score=settings.kb_min_score,
            metadata=metadata,
        )

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[dict[str, Any]]:
        """Return the passages that answer `query`.

        Args:
            query: the customer's question, or a rewritten stand-in for it.
            top_k: how many passages to consider, defaulting to the configured value.
            min_score: cosine similarity floor, defaulting to the configured value.

        Returns:
            `{"text", "source", "heading", "citation", "score"}` dicts, best
            first. Empty when nothing clears the floor, which tells the agent to
            say it does not know rather than guess.
        """
        query = (query or "").strip()
        if not query or not self.passages:
            return []

        limit = min(top_k or self.top_k, len(self.passages))
        vector = self.embedder.embed_query(query)
        if not vector:
            return []

        import numpy as np

        scores, indices = self.index.search(np.asarray([vector], dtype="float32"), limit)
        floor = self.min_score if min_score is None else min_score

        results: list[dict[str, Any]] = []
        for score, position in zip(scores[0], indices[0]):
            if position < 0 or score < floor:
                # Sorted descending, so nothing better can follow.
                break
            passage = self.passages[int(position)]
            results.append(
                {
                    "text": passage.text,
                    "source": passage.source,
                    "heading": passage.heading,
                    "citation": passage.citation,
                    "score": round(float(score), 4),
                }
            )
        return results

    def search_as_context(self, query: str, **kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        """Search and flatten the passages into one context string.

        Returns:
            `(context, passages)`. The context carries the citation above each
            passage so the model can quote it, and is empty when nothing passed
            the floor.
        """
        passages = self.search(query, **kwargs)
        if not passages:
            return "", []
        context = "\n\n".join(
            f"{passage['citation']}\n{passage['text']}" for passage in passages
        )
        return context, passages

    def health(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "passages": len(self.passages),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "embedding_model": self.metadata.get("model", "unknown"),
        }


_retriever: Retriever | None = None
_lock = threading.Lock()


def get_retriever() -> Retriever:
    """The shared retriever, loaded on first use."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = Retriever.load()
    return _retriever


def set_retriever(retriever: Retriever | None) -> None:
    """Replace the shared retriever. Used by tests."""
    global _retriever
    _retriever = retriever


def is_index_built(settings: Settings | None = None) -> bool:
    """Whether `scripts/build_kb.py` has been run."""
    settings = settings or get_settings()
    return (settings.kb_index_dir / "index.faiss").exists()
