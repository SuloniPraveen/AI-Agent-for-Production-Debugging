#!/usr/bin/env python3
"""Chunk text files, embed with OpenAI, and insert into rag_chunk for agent retrieval.

Examples (from repo root, with `.env.development` and DB running):

  PYTHONPATH=. uv run python scripts/ingest_rag_documents.py \\
    --file ./docs/runbook.md --doc-type runbook --source-label runbook.md

  PYTHONPATH=. uv run python scripts/ingest_rag_documents.py \\
    --file ./sample.log --doc-type log --source-label payment-service
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 180) -> list[str]:
    """Split text into overlapping windows for embedding."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return pieces


async def _ingest(path: Path, doc_type: str, source_label: str, batch_size: int) -> int:
    from app.models.rag_chunk import RagChunk
    from app.services.rag import rag_service

    raw = path.read_text(encoding="utf-8")
    chunks = chunk_text(raw)
    if not chunks:
        print("No text to ingest.", file=sys.stderr)
        return 0

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = await rag_service.embed_texts(batch)
        rows = [
            RagChunk(doc_type=doc_type, source_label=source_label, content=content, embedding=emb)
            for content, emb in zip(batch, embeddings, strict=True)
        ]
        total += await rag_service.insert_chunks(rows)
        print(f"Ingested {total} / {len(chunks)} chunks...", flush=True)
    return total


def main() -> None:
    load_dotenv(".env.development")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest documents into rag_chunk (pgvector).")
    parser.add_argument("--file", type=Path, required=True, help="Path to a UTF-8 text or markdown file.")
    parser.add_argument(
        "--doc-type",
        choices=("log", "runbook"),
        required=True,
        help="Chunk category: log lines vs runbook/procedure text.",
    )
    parser.add_argument(
        "--source-label",
        default="",
        help="Display name (defaults to file name).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default 32).")
    args = parser.parse_args()

    path = args.file.expanduser().resolve()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    label = args.source_label or path.name

    count = asyncio.run(_ingest(path, args.doc_type, label, max(1, min(args.batch_size, 64))))
    print(f"Done. Inserted {count} chunk(s) into rag_chunk.")


if __name__ == "__main__":
    main()
