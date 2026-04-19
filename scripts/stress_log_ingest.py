#!/usr/bin/env python3
"""Generate a large sample log file and ingest via the sync pipeline (Phase 1 exit check).

Creates ≥50k lines, writes a temp file, runs the same ingestion as POST /upload background task.

  PYTHONPATH=. uv run python scripts/stress_log_ingest.py

Requires: Postgres, OPENAI_API_KEY, `.env.development` (or env) matching the app DB.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


def main() -> None:
    load_dotenv(".env.development")
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=50_000, help="Number of lines (default 50000)")
    args = parser.parse_args()

    from sqlmodel import Session, select

    from app.models.user import User
    from app.services.database import database_service
    from app.services.log_ingestion import create_pending_batch, run_ingestion_sync

    with Session(database_service.engine) as session:
        user = session.exec(select(User)).first()
    if user is None:
        print("No user in DB — register one via /api/v1/auth/register first.", file=sys.stderr)
        sys.exit(1)

    n = max(1, args.lines)
    tmp = Path(tempfile.mkdtemp()) / "sample.log"
    # Mix plain and JSON-shaped lines for metadata extraction
    with tmp.open("w", encoding="utf-8") as f:
        for i in range(n):
            if i % 500 == 0:
                f.write(
                    '{"timestamp":"2025-06-01T12:00:00Z","level":"ERROR","service":"payments","msg":"timeout"}\n'
                )
            else:
                f.write(f"2025-06-01T12:00:01Z INFO checkout order={i} ok\n")

    batch = create_pending_batch(user_id=user.id, filename="stress.log")
    print(f"batch_id={batch.id} lines={n} file={tmp}")
    run_ingestion_sync(batch.id, str(tmp))
    print("done — check log_batches / log_chunks in Postgres or GET /api/v1/logs/batches/{id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
