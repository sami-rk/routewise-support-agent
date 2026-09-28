"""Tests for heading-aware chunking.

The point of the chunker is that a passage carries the heading it came from, so
a table is never cut in half and the citation is meaningful.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.chunking import (
    Chunk,
    chunk_directory,
    chunk_markdown,
    load_documents,
    split_long_text,
    split_sections,
)

SAMPLE = """# CloudSync Pro

Intro paragraph about the product.

## The plans

| Plan | Price |
|---|---|
| Basic | $9/mo |
| Pro | $19/mo |

### Business

Business is $49 per month.

## Cancelling

Cancel from Account > Subscription.
"""


def test_splits_on_headings() -> None:
    headings = [heading for heading, _body in split_sections(SAMPLE)]
    assert headings == ["CloudSync Pro", "The plans", "Business", "Cancelling"]


def test_text_before_the_first_heading_is_kept() -> None:
    text = "Just a paragraph, no heading at all.\n\nAnd another."
    sections = split_sections(text)
    assert sections[0][0] == ""
    assert "Just a paragraph" in sections[0][1]


def test_each_chunk_carries_its_heading() -> None:
    chunks = chunk_markdown(SAMPLE, "pricing.md")
    headings = {chunk.heading for chunk in chunks}
    assert "The plans" in headings
    assert "Business" in headings
    assert "Cancelling" in headings


def test_chunk_text_includes_its_heading() -> None:
    for chunk in chunk_markdown(SAMPLE, "pricing.md"):
        if chunk.heading:
            assert chunk.text.startswith(chunk.heading)


def test_nested_headings_are_kept_separate() -> None:
    chunks = chunk_markdown(SAMPLE, "pricing.md")
    business = next(c for c in chunks if c.heading == "Business")
    assert "$49 per month" in business.text
    # The parent's table did not bleed into the child.
    assert "$19/mo" not in business.text


def test_table_is_not_cut_up_when_it_fits() -> None:
    chunks = chunk_markdown(SAMPLE, "pricing.md")
    plans = next(c for c in chunks if c.heading == "The plans")
    assert "Pro | $19/mo" in plans.text


def test_source_is_recorded_on_every_chunk() -> None:
    for chunk in chunk_markdown(SAMPLE, "pricing.md"):
        assert chunk.source == "pricing.md"


def test_citation_format() -> None:
    assert chunk_markdown(SAMPLE, "pricing.md")[0].citation == "[pricing.md › CloudSync Pro]"


def test_citation_without_a_heading() -> None:
    chunk = Chunk(text="body", source="notes.txt", heading="")
    assert chunk.citation == "[notes.txt]"


def test_no_empty_chunks() -> None:
    for chunk in chunk_markdown("# A\n\n## B\n\n### C\n", "empty.md"):
        assert chunk.text.strip()


def test_long_section_is_split_and_keeps_overlap() -> None:
    body = "\n\n".join(f"Paragraph number {i} with enough text to matter." for i in range(60))
    pieces = split_long_text(body, max_chars=600, overlap=100)
    assert len(pieces) > 1
    assert all(len(piece) <= 600 for piece in pieces)
    # Consecutive passages share text, so a query spanning the split still hits.
    assert any(pieces[i][-50:] in pieces[i + 1] for i in range(len(pieces) - 1))


def test_one_very_long_sentence_is_still_cut() -> None:
    pieces = split_long_text("word " * 2000, max_chars=500, overlap=0)
    assert all(len(piece) <= 500 for piece in pieces)
    assert len(pieces) > 1


def test_short_text_is_not_split() -> None:
    assert split_long_text("A short body.", max_chars=500, overlap=0) == ["A short body."]


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_markdown("body", "x.md", max_chars=100, overlap=100)
    with pytest.raises(ValueError, match="max_chars"):
        chunk_markdown("body", "x.md", max_chars=0)


def test_load_documents_reads_md_and_txt(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("ignored", encoding="utf-8")
    sources = [name for name, _ in load_documents(tmp_path)]
    assert sources == ["a.md", "b.txt"]


def test_load_documents_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_chunk_directory_covers_every_document(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\n## C\n\nSome body text here.", encoding="utf-8")
    chunks = chunk_directory(tmp_path)
    assert {chunk.source for chunk in chunks} == {"a.md", "b.md"}


def test_real_knowledge_base_chunks_cleanly() -> None:
    kb_dir = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
    chunks = chunk_directory(kb_dir)
    assert len(chunks) > 8
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.citation for chunk in chunks)
