#!/usr/bin/env python3
"""Write demo_data/sample_service.log with N synthetic lines (default 50_000)."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "demo_data" / "sample_service.log"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lines", type=int, default=50_000)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
    services = ("checkout", "payment", "gateway", "auth", "inventory")
    n = args.lines

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(n):
            ts = start + timedelta(milliseconds=50 * i)
            svc = services[i % len(services)]
            rid = f"req-{i:06x}"
            if 24_000 <= i <= 24_200:
                line = (
                    f"{ts.isoformat()} {svc} ERROR "
                    f"request_id={rid} "
                    f"msg=\"payment-db connection refused host=payment-db.internal port=5432 attempt={i}\""
                )
            elif 31_000 <= i <= 31_080:
                line = (
                    f"{ts.isoformat()} gateway WARN "
                    f"request_id={rid} "
                    f"msg=\"upstream vendor_fraud_api latency_ms=28000 timeout_threshold=25000\""
                )
            elif 38_500 <= i <= 38_520:
                line = (
                    f"{ts.isoformat()} checkout ERROR "
                    f"request_id={rid} "
                    f"msg=\"redis cache MISS storm key_prefix=cart session_rebuild backlog=1200\""
                )
            else:
                lvl = "INFO" if i % 17 else "WARN"
                line = (
                    f"{ts.isoformat()} {svc} {lvl} "
                    f"request_id={rid} "
                    f"msg=\"handler=/{svc}/v1 ok latency_ms={10 + (i % 80)}\""
                )
            f.write(line + "\n")

    print(f"Wrote {n} lines to {args.out}")


if __name__ == "__main__":
    main()
