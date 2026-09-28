"""Build and persist the FAISS index over the knowledge base.

The original rebuilt its in-memory index from paid OpenAI embeddings on every
start. Here the index is built once, with local embeddings, and written to disk
as two files: the FAISS index itself and a JSON sidecar holding the passage
metadata, kept in step by chunk order.

Vectors are L2-normalised and the index is `IndexFlatIP`, so a search score is a
cosine similarity in 0..1 and `KB_MIN_SCORE` reads directly against it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk, chunk_directory
from .ingest import Embedder

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "chunks.json"


@dataclass
class Passage:
    """One indexed chunk, with the text and its citation."""

    text: str
    source: str
    heading: str

    @property
    def citation(self) -> str:
        if self.heading:
            return f"[{self.source} › {self.heading}]"
        return f"[{self.source}]"

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Passage":
        return cls(text=chunk.text, source=chunk.source, heading=chunk.heading)


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    index_path: Path,
    *,
    metadata_path: Path | None = None,
) -> int:
    """Embed chunks and write the index and its metadata to disk.

    Args:
        chunks: passages to index, in the order they should be stored.
        embedder: the local embedder.
        index_path: where the FAISS index goes.
        metadata_path: the JSON sidecar; defaults to `chunks.json` beside the
            index.

    Returns:
        The number of passages indexed.

    Raises:
        ValueError: if there are no chunks, or the embedder returns a different
            width for two passages, which would corrupt the index.
    """
    if not chunks:
        raise ValueError("No passages to index; is the knowledge base empty?")

    import faiss

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedder returned a different number of vectors than passages")

    widths = {len(vector) for vector in vectors}
    if len(widths) != 1:
        raise ValueError(f"Embedder returned inconsistent vector widths: {sorted(widths)}")
    width = widths.pop()

    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(width)  # inner product == cosine, vectors are normalised
    index.add(matrix)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    metadata_path = metadata_path or index_path.with_name(METADATA_FILENAME)
    payload = {
        "model": getattr(embedder, "model_name", "unknown"),
        "dimension": width,
        "count": len(chunks),
        "chunks": [asdict(Passage.from_chunk(chunk)) for chunk in chunks],
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(chunks)


def load_index(index_path: Path, metadata_path: Path | None = None) -> tuple[Any, list[Passage], dict[str, Any]]:
    """Load a persisted index and its passages.

    Args:
        index_path: the FAISS index.
        metadata_path: the JSON sidecar; defaults to `chunks.json` beside it.

    Returns:
        `(faiss_index, passages, metadata)`.

    Raises:
        FileNotFoundError: if the index has not been built yet.
    """
    import faiss

    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index at {index_path}. Build it first: python -m scripts.build_kb"
        )
    metadata_path = metadata_path or index_path.with_name(METADATA_FILENAME)
    if not metadata_path.exists():
        raise FileNotFoundError(f"No chunk metadata at {metadata_path}; rebuild the index.")

    index = faiss.read_index(str(index_path))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    passages = [Passage(**chunk) for chunk in payload["chunks"]]

    if index.ntotal != len(passages):
        raise ValueError(
            f"Index holds {index.ntotal} vectors but metadata lists {len(passages)} passages; "
            "the two files are out of step, so rebuild the index."
        )
    return index, passages, {k: v for k, v in payload.items() if k != "chunks"}


def build_from_directory(
    kb_dir: Path,
    embedder: Embedder,
    index_path: Path,
    *,
    max_chars: int = 1800,
    overlap: int = 200,
) -> int:
    """Chunk a knowledge base directory and index it in one step."""
    chunks = chunk_directory(kb_dir, max_chars=max_chars, overlap=overlap)
    return build_index(chunks, embedder, index_path)
