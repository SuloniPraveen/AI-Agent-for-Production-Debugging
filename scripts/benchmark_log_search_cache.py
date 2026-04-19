#!/usr/bin/env python3
"""Measure log search latency: cold (unique queries, cache miss) vs warm (repeat, cache hit).

Requires: Postgres, optional Redis (`REDIS_URL`), `OPENAI_API_KEY`, and at least one row in `log_chunks` (ingest a log file first).

Writes JSON summary to evals/reports/log_search_cache_benchmark_<ts>.json

  REDIS_URL=redis://localhost:6379/0 PYTHONPATH=. uv run python scripts/benchmark_log_search_cache.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def _pctl(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[i]


async def _run() -> dict:
    from sqlmodel import Session, select

    from app.core.config import settings
    from app.models.log_chunk import LogChunk
    from app.services.database import database_service
    from app.services.log_search import log_search_service
    from app.services.log_search_scope import (
        reset_log_search_batch_ids,
        set_log_search_batch_ids,
    )

    with Session(database_service.engine) as session:
        one = session.exec(select(LogChunk).limit(1)).first()
    if one is None:
        return {
            "error": "no_log_chunks",
            "note": "Ingest at least one log file before running this benchmark.",
        }

    scope_token = set_log_search_batch_ids((str(one.batch_id),))

    base = "benchmark log search latency probe"
    cold_n = 25
    warm_n = 50

    cold: list[float] = []
    try:
        for i in range(cold_n):
            t0 = time.perf_counter()
            await log_search_service.search_json(f"{base} unique={i}")
            cold.append(time.perf_counter() - t0)

        warm_query = f"{base} repeated"
        # prime once
        await log_search_service.search_json(warm_query)
        warm: list[float] = []
        for _ in range(warm_n):
            t0 = time.perf_counter()
            await log_search_service.search_json(warm_query)
            warm.append(time.perf_counter() - t0)
    finally:
        reset_log_search_batch_ids(scope_token)

    cold_s = sorted(cold)
    warm_s = sorted(warm)
    cold_p50 = _pctl(cold_s, 50)
    cold_p95 = _pctl(cold_s, 95)
    warm_p50 = _pctl(warm_s, 50)
    warm_p95 = _pctl(warm_s, 95)
    reduction_p95 = (
        ((cold_p95 - warm_p95) / cold_p95 * 100.0) if cold_p95 > 1e-9 else 0.0
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "redis_url_set": bool(settings.REDIS_URL),
        "cache_ttl_seconds": settings.LOG_SEARCH_CACHE_TTL_SECONDS,
        "cold_queries": cold_n,
        "warm_repeats": warm_n,
        "cold_ms": {"p50": cold_p50 * 1000, "p95": cold_p95 * 1000},
        "warm_ms": {"p50": warm_p50 * 1000, "p95": warm_p95 * 1000},
        "approx_latency_reduction_p95_pct": round(reduction_p95, 2),
        "note": "Warm path assumes Redis hit after first repeat; without REDIS_URL, warm ~= cold.",
    }


def main() -> None:
    load_dotenv(".env.development")
    load_dotenv()

    out = asyncio.run(_run())
    reports = Path(__file__).resolve().parent.parent / "evals" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = reports / f"log_search_cache_benchmark_{ts}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
