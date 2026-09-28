"""Split knowledge base documents into retrievable passages.

The original project used 200-character chunks with no awareness of structure,
which cut tables in half and lost the heading that gave a passage its meaning.
Here a document is split on its headings first, so each passage keeps the section
it came from, and only over-long sections are broken down further.

Passages are roughly 300 to 500 tokens with a small overlap, and each carries
`source` and `heading` for the citation the customer sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Headings are kept as their own line in the passage, which is what the
# retriever quotes back as `[file.md › Section]`.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """One retrievable passage."""

    text: str
    source: str
    heading: str

    @property
    def citation(self) -> str:
        """`[source.md › Heading]`, as quoted in an answer."""
        if self.heading:
            return f"[{self.source} › {self.heading}]"
        return f"[{self.source}]"


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into `(heading, body)` pairs.

    The text before the first heading is kept under an empty heading, so a
    document that opens with a paragraph rather than a title is not dropped.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if buffer and any(part.strip() for part in buffer):
                sections.append((heading, "\n".join(buffer).strip()))
            heading = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)

    if buffer and any(part.strip() for part in buffer):
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Break an over-long section on paragraph, then sentence, then word.

    Args:
        text: the section body.
        max_chars: soft ceiling for one passage.
        overlap: characters repeated between neighbouring passages, so a
            sentence spanning a split is still found by a query.

    Returns:
        Passages, each at most `max_chars` long unless a single word is longer.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for paragraph in re.split(r"\n\s*\n", text):
        candidate = paragraph.strip()
        if not candidate:
            continue
        # A single paragraph can still be too long on its own.
        if len(candidate) > max_chars:
            if current:
                pieces.append("\n\n".join(current))
                current, length = [], 0
            pieces.extend(_split_sentences(candidate, max_chars, overlap))
            continue
        if length + len(candidate) + 2 > max_chars and current:
            pieces.append("\n\n".join(current))
            # Carry the tail of the previous passage into this one.
            tail = "\n\n".join(current)[-overlap:].strip() if overlap else ""
            current = [tail, candidate] if tail else [candidate]
            length = sum(len(part) for part in current) + 2
        else:
            current.append(candidate)
            length += len(candidate) + 2

    if current:
        pieces.append("\n\n".join(current).strip())
    return [piece for piece in pieces if piece.strip()]


def _split_sentences(text: str, max_chars: int, overlap: int) -> list[str]:
    """Last-resort split on sentence boundaries, then on words."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)

    # A single sentence longer than the ceiling still has to be cut.
    result: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            result.append(piece[:max_chars])
            piece = piece[max_chars - overlap :] if overlap else piece[max_chars:]
        if piece.strip():
            result.append(piece)
    return result


def chunk_markdown(text: str, source: str, *, max_chars: int = 1800, overlap: int = 200) -> list[Chunk]:
    """Turn one markdown document into passages.

    Args:
        text: the document body.
        source: the file name, used for the citation.
        max_chars: soft ceiling per passage. 1800 characters is roughly 400
            tokens, inside the 300-500 target.
        overlap: characters repeated across a split.

    Returns:
        The passages, in document order. Each begins with its heading, so the
        text alone carries enough context to answer from.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")

    chunks: list[Chunk] = []
    for heading, body in split_sections(text):
        for piece in split_long_text(body, max_chars, overlap):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    text=f"{heading}\n{piece}".strip() if heading else piece.strip(),
                    source=source,
                    heading=heading,
                )
            )
    return chunks


def load_documents(kb_dir: Path) -> list[tuple[str, str]]:
    """Read `.md` and `.txt` files from a directory, sorted by name.

    Args:
        kb_dir: the knowledge base directory.

    Returns:
        `(filename, contents)` pairs.

    Raises:
        FileNotFoundError: if the directory does not exist.
    """
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    documents: list[tuple[str, str]] = []
    for path in sorted(kb_dir.iterdir()):
        if path.suffix.lower() not in (".md", ".txt") or not path.is_file():
            continue
        documents.append((path.name, path.read_text(encoding="utf-8")))
    return documents


def chunk_directory(kb_dir: Path, **kwargs: int) -> list[Chunk]:
    """Chunk every document in a directory."""
    chunks: list[Chunk] = []
    for source, text in load_documents(kb_dir):
        chunks.extend(chunk_markdown(text, source, **kwargs))  # type: ignore[arg-type]
    return chunks
