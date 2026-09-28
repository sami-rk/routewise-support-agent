"""Build the FAISS index over a knowledge base directory.

    python -m scripts.build_kb
    python -m scripts.build_kb --kb-dir ./docs          # your own .md/.txt files
    python -m scripts.build_kb --dry-run                # chunk and report, build nothing

Embeddings are local, so this needs no API key. The index and a JSON sidecar of
passage metadata are written to `KB_DIR`'s sibling `kb_index/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings, validate_settings
from app.rag.chunking import chunk_directory
from app.rag.indexer import INDEX_FILENAME, build_index
from app.rag.ingest import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder


def main() -> None:
    # Refuse a non-free LLM_MODELS before spending time embedding.
    settings = validate_settings()
    parser = argparse.ArgumentParser(description="Build the FAISS knowledge base index.")
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=settings.kb_dir,
        help="Directory of .md/.txt documents (default: KB_DIR).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.kb_index_dir,
        help="Where to write the index (default: KB_DIR's sibling kb_index).",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--max-chars", type=int, default=1800, help="Passage size ceiling.")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between split passages.")
    parser.add_argument("--dry-run", action="store_true", help="Report the passages, build nothing.")
    args = parser.parse_args()

    kb_dir = args.kb_dir if args.kb_dir.is_absolute() else (Path.cwd() / args.kb_dir).resolve()
    chunks = chunk_directory(kb_dir, max_chars=args.max_chars, overlap=args.overlap)

    sources: dict[str, int] = {}
    for chunk in chunks:
        sources[chunk.source] = sources.get(chunk.source, 0) + 1

    print(f"{kb_dir}: {len(sources)} documents, {len(chunks)} passages")
    for source, count in sorted(sources.items()):
        print(f"  {source:<32} {count:>3} passages")

    if args.dry_run:
        print("\n--dry-run: index not written")
        for chunk in chunks[:5]:
            print(f"  {chunk.citation}\n    {chunk.text[:90]}...")
        return

    if not chunks:
        raise SystemExit(f"No .md or .txt documents found in {kb_dir}")

    index_path = args.index_dir / INDEX_FILENAME
    embedder = SentenceTransformerEmbedder(args.embedding_model)
    count = build_index(
        chunks, embedder, index_path, metadata_path=args.index_dir / "chunks.json"
    )
    print(f"\nindexed {count} passages with {args.embedding_model}")
    print(f"wrote {index_path} and {args.index_dir / 'chunks.json'}")


if __name__ == "__main__":
    main()
