"""Tests for the FAISS index and the retriever.

The index is built with `HashEmbedder`, so these tests need no model download
and run offline. Retrieval is verified on structure (shape, ordering, floor,
citations) and on a hand-built index where the right answer is known.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.chunking import chunk_markdown
from app.rag.indexer import (
    Passage,
    build_from_directory,
    build_index,
    load_index,
)
from app.rag.ingest import HashEmbedder
from app.rag.retriever import Retriever, is_index_built

DOC = """# Refunds

A charge is refundable within 14 days of the charge date.

## Large refunds

Refunds above $20 need a specialist to approve.
"""


@pytest.fixture
def built(tmp_path: Path) -> tuple[Retriever, Path]:
    """A two-passage index built from a small document, in a temp directory."""
    index_path = tmp_path / "index.faiss"
    chunks = chunk_markdown(DOC, "cancellation_refunds.md", max_chars=400, overlap=0)
    build_index(chunks, HashEmbedder(64), index_path)
    retriever = Retriever(
        *load_index(index_path)[:2], HashEmbedder(64), top_k=4, min_score=-1.0
    )
    return retriever, index_path


class TestIndex:
    def test_index_and_metadata_are_written(self, tmp_path: Path) -> None:
        index_path = tmp_path / "kb" / "index.faiss"
        count = build_index(chunk_markdown(DOC, "doc.md"), HashEmbedder(64), index_path)
        assert index_path.exists()
        assert index_path.with_name("chunks.json").exists()
        assert count >= 1

    def test_index_survives_a_round_trip(self, tmp_path: Path) -> None:
        index_path = tmp_path / "index.faiss"
        chunks = chunk_markdown(DOC, "doc.md")
        build_index(chunks, HashEmbedder(64), index_path)
        index, passages, metadata = load_index(index_path)
        assert index.ntotal == len(chunks) == len(passages)
        assert metadata["dimension"] == 64
        assert passages[0].source == "doc.md"

    def test_passages_keep_text_and_citation(self, tmp_path: Path) -> None:
        index_path = tmp_path / "index.faiss"
        build_index(chunk_markdown(DOC, "doc.md"), HashEmbedder(64), index_path)
        _index, passages, _meta = load_index(index_path)
        joined = " ".join(p.text for p in passages)
        assert "14 days" in joined
        assert any(p.citation.startswith("[doc.md › ") for p in passages)

    def test_empty_knowledge_base_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No passages"):
            build_index([], HashEmbedder(64), tmp_path / "index.faiss")

    def test_missing_index_says_how_to_build_it(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="build_kb"):
            load_index(tmp_path / "nope.faiss")

    def test_index_and_metadata_out_of_step_is_detected(self, tmp_path: Path) -> None:
        index_path = tmp_path / "index.faiss"
        build_index(chunk_markdown(DOC, "doc.md"), HashEmbedder(64), index_path)
        # Simulate a stale sidecar by rewriting it with fewer passages.
        metadata_path = index_path.with_name("chunks.json")
        metadata_path.write_text('{"model": "hash", "dimension": 64, "chunks": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="out of step"):
            load_index(index_path)

    def test_build_from_directory(self, tmp_path: Path) -> None:
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "a.md").write_text(DOC, encoding="utf-8")
        (kb / "b.md").write_text("# Sync\n\nSync keeps files current.", encoding="utf-8")
        count = build_from_directory(kb, HashEmbedder(64), tmp_path / "index.faiss")
        assert count >= 2


class TestRetriever:
    def test_returns_passages_with_citations(self, built) -> None:
        retriever, _ = built
        results = retriever.search("refunds within 14 days")
        assert results
        for result in results:
            assert set(result) == {"text", "source", "heading", "citation", "score"}
            assert result["citation"].startswith("[cancellation_refunds.md")
            assert 0.0 <= result["score"] <= 1.0

    def test_best_match_comes_first(self, built) -> None:
        retriever, _ = built
        results = retriever.search("refunds above 20 dollars need a specialist")
        assert results[0]["heading"] == "Large refunds"

    def test_similarity_floor_can_reject_everything(self, built) -> None:
        retriever, _ = built
        assert retriever.search("refunds", min_score=1.01) == []

    def test_low_floor_returns_results(self, built) -> None:
        retriever, _ = built
        assert retriever.search("something unrelated to this document", min_score=-1.0)

    def test_top_k_is_respected(self, built) -> None:
        retriever, _ = built
        assert len(retriever.search("refund", top_k=1, min_score=-1.0)) == 1

    def test_empty_query_returns_nothing(self, built) -> None:
        retriever, _ = built
        assert retriever.search("") == []
        assert retriever.search("   ") == []

    def test_context_string_carries_citations(self, built) -> None:
        retriever, _ = built
        context, passages = retriever.search_as_context("refund policy", min_score=-1.0)
        assert passages
        assert context.count("[cancellation_refunds.md") == len(passages)
        assert "14 days" in context

    def test_context_is_empty_when_nothing_matches(self, built) -> None:
        retriever, _ = built
        context, passages = retriever.search_as_context("refund", min_score=1.01)
        assert context == ""
        assert passages == []

    def test_health_reports_passage_count(self, built) -> None:
        retriever, _ = built
        health = retriever.health()
        assert health["loaded"] is True
        assert health["passages"] == len(retriever.passages)

    def test_is_index_built(self, tmp_path: Path) -> None:
        from app.config import Settings

        settings = Settings(_env_file=None, kb_dir=tmp_path / "kb")
        assert is_index_built(settings) is False
        (tmp_path / "kb_index").mkdir(parents=True, exist_ok=True)
        (tmp_path / "kb_index" / "index.faiss").write_bytes(b"")
        assert is_index_built(settings) is True

    def test_passage_citation_helper(self) -> None:
        assert Passage("t", "a.md", "H").citation == "[a.md › H]"
        assert Passage("t", "a.md", "").citation == "[a.md]"
