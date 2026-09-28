"""Local sentence-transformers embedder.

Embeddings run on this machine, never through a paid API. The index is built
once with `scripts/build_kb.py` and reloaded at startup, so the model is only
loaded when a build actually happens or a retriever is constructed.

`bge-small-en-v1.5` is used because it is small (about 130 MB), fast on CPU and
strong enough for a few hundred support passages. Vectors are L2-normalised, so
an inner product is a cosine similarity in 0..1 — which is what `KB_MIN_SCORE`
is compared against.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, Sequence, runtime_checkable

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into fixed-width vectors."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for the index."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query in the same space."""
        ...

    @property
    def dimension(self) -> int:
        """Length of one vector."""
        ...


class SentenceTransformerEmbedder:
    """`Embedder` backed by a local sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Any:
        # Loading twice from two threads would double the memory and the wait.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_loaded()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,  # cosine similarity == inner product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []

    @property
    def dimension(self) -> int:
        model = self._ensure_loaded()
        return int(model.get_sentence_embedding_dimension())

    def health(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "loaded": self._model is not None,
            "device": self.device,
        }


class HashEmbedder:
    """A deterministic offline embedder, for tests.

    Counts word tokens into hashed buckets and L2-normalises, so the inner
    product is a non-negative lexical overlap in 0..1. It has no semantic
    understanding — "refund" and "money back" are unrelated to it — but text
    that shares words scores above text that does not, which is what the
    retrieval plumbing needs in order to be testable offline.

    Do not use this to judge retrieval quality; use the real embedding model.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        # Named so `Retriever.health()` and the built index say what they used
        # rather than "unknown".
        self.model_name = f"hash-{dimension}"

    def _vector(self, text: str) -> list[float]:
        import hashlib
        import math
        import re

        vector = [0.0] * self._dimension
        # Word characters only: splitting a markdown table on whitespace makes
        # `|` and `---` count as terms, which drowns out the real overlap.
        for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return self._dimension

    def health(self) -> dict[str, Any]:
        return {"model": "hash", "loaded": True, "device": "cpu"}
