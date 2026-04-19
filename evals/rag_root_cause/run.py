"""Reproducible root-cause eval: three baselines, recall@k, JSON report.

Run:
  make eval-rag
  # or
  ENV=development PYTHONPATH=. uv run python -m evals.rag_root_cause.run

Requires OPENAI_API_KEY (or EVALUATION_API_KEY). See evals/rag_root_cause/METHODOLOGY.md.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from openai import AsyncOpenAI

# Repo root on path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import settings

DATASET_PATH = Path(__file__).parent / "dataset.yaml"
REPORTS_DIR = _ROOT / "evals" / "reports"
K_RETRIEVAL = 5


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _naive_rank_ids(query: str, chunks: list[dict]) -> list[str]:
    qtokens = {t for t in query.lower().split() if len(t) > 2}
    scored: list[tuple[int, str]] = []
    for c in chunks:
        text = c["text"].lower()
        score = sum(1 for t in qtokens if t in text)
        scored.append((score, c["id"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in scored]


def _recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    top = set(ranked_ids[:k])
    gold = set(gold_ids)
    return len(top & gold) / len(gold)


def _load_dataset(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _vocabulary(incidents: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for inc in incidents:
        s = inc["gold_root_cause"]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)


async def _embed(client: AsyncOpenAI, model: str, text: str, dimensions: int) -> list[float]:
    kwargs: dict[str, Any] = {"model": model, "input": text}
    if model.startswith("text-embedding-3") and dimensions != 1536:
        kwargs["dimensions"] = dimensions
    r = await client.embeddings.create(**kwargs)
    return list(r.data[0].embedding)


async def _embed_many(
    client: AsyncOpenAI, model: str, texts: list[str], dimensions: int, batch_size: int = 32
) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        kwargs: dict[str, Any] = {"model": model, "input": batch}
        if model.startswith("text-embedding-3") and dimensions != 1536:
            kwargs["dimensions"] = dimensions
        r = await client.embeddings.create(**kwargs)
        by_i = {d.index: list(d.embedding) for d in r.data}
        out.extend(by_i[j] for j in range(len(batch)))
    return out


def _rag_rank_ids(query_emb: list[float], chunks: list[dict], chunk_embs: list[list[float]]) -> list[str]:
    sims = [_cosine(query_emb, e) for e in chunk_embs]
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    return [chunks[i]["id"] for i in order]


async def _llm_predict(
    client: AsyncOpenAI,
    judge_model: str,
    incident_description: str,
    slug_list: list[str],
    context: Optional[str],
    mode_label: str,
) -> dict[str, Any]:
    slug_lines = "\n".join(f"- {s}" for s in slug_list)
    if context:
        body = (
            f"Incident:\n{incident_description}\n\n"
            f"Allowed root_cause slugs (choose only from this list):\n{slug_lines}\n\n"
            f"Evidence log excerpts (may be noisy):\n{context}\n\n"
            "Return JSON with keys: root_cause (one slug from the list), "
            "alternatives (array of exactly two other slugs from the list for runner-up guesses). "
            "Use slug strings exactly as listed."
        )
    else:
        body = (
            f"Incident:\n{incident_description}\n\n"
            f"Allowed root_cause slugs (choose only from this list):\n{slug_lines}\n\n"
            "No log excerpts provided. Infer from the incident text only.\n\n"
            "Return JSON with keys: root_cause (one slug from the list), "
            "alternatives (array of exactly two other slugs from the list)."
        )

    resp = await client.chat.completions.create(
        model=judge_model,
        messages=[
            {
                "role": "system",
                "content": "You are a production incident analyst. Output valid JSON only.",
            },
            {"role": "user", "content": body},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"root_cause": "", "alternatives": [], "parse_error": True, "mode": mode_label}


def _score_prediction(pred: dict, gold: str) -> tuple[bool, bool]:
    rc = (pred.get("root_cause") or "").strip()
    alts = pred.get("alternatives") or []
    if not isinstance(alts, list):
        alts = []
    top3 = [rc] + [str(x).strip() for x in alts[:2]]
    exact = rc == gold
    top3_hit = gold in top3
    return exact, top3_hit


async def _eval_one_incident(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    embed_model: str,
    embed_dim: int,
    judge_model: str,
    slug_list: list[str],
    inc: dict,
) -> dict[str, Any]:
    async with sem:
        q = inc["description"]
        gold = inc["gold_root_cause"]
        gold_ids = list(inc["gold_evidence_chunk_ids"])
        chunks = list(inc["chunks"])

        naive_ids = _naive_rank_ids(q, chunks)
        recall_naive = _recall_at_k(naive_ids, gold_ids, K_RETRIEVAL)

        texts = [c["text"] for c in chunks]
        q_emb = await _embed(client, embed_model, q, embed_dim)
        chunk_embs = await _embed_many(client, embed_model, texts, embed_dim)
        rag_ids = _rag_rank_ids(q_emb, chunks, chunk_embs)
        recall_rag = _recall_at_k(rag_ids, gold_ids, K_RETRIEVAL)

        top_naive = naive_ids[:K_RETRIEVAL]
        top_rag = rag_ids[:K_RETRIEVAL]
        ctx_naive = "\n---\n".join(next(c["text"] for c in chunks if c["id"] == i) for i in top_naive)
        ctx_rag = "\n---\n".join(next(c["text"] for c in chunks if c["id"] == i) for i in top_rag)

        pred_none = await _llm_predict(client, judge_model, q, slug_list, None, "llm_no_retrieval")
        ex0, t30 = _score_prediction(pred_none, gold)

        pred_naive = await _llm_predict(client, judge_model, q, slug_list, ctx_naive, "naive_fulltext")
        ex1, t31 = _score_prediction(pred_naive, gold)

        pred_rag = await _llm_predict(client, judge_model, q, slug_list, ctx_rag, "rag_embedding")
        ex2, t32 = _score_prediction(pred_rag, gold)

        return {
            "incident_id": inc["id"],
            "gold_root_cause": gold,
            "recall_at_5": {
                "naive_fulltext": recall_naive,
                "rag_embedding": recall_rag,
            },
            "root_cause_exact": {
                "llm_no_retrieval": ex0,
                "naive_fulltext": ex1,
                "rag_embedding": ex2,
            },
            "root_cause_top3": {
                "llm_no_retrieval": t30,
                "naive_fulltext": t31,
                "rag_embedding": t32,
            },
            "human_rubric_sample": bool(inc.get("human_rubric_sample", False)),
            "predictions": {
                "llm_no_retrieval": pred_none,
                "naive_fulltext": pred_naive,
                "rag_embedding": pred_rag,
            },
        }


def _mean(xs: list[bool]) -> float:
    return sum(1 for x in xs if x) / len(xs) if xs else 0.0


def _mean_f(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


async def async_main() -> dict[str, Any]:
    api_key = settings.OPENAI_API_KEY or os.getenv("EVALUATION_API_KEY", "")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or EVALUATION_API_KEY for eval-rag.")

    judge_model = os.getenv("EVAL_RAG_JUDGE_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("EVAL_RAG_EMBED_MODEL", settings.LOG_EMBEDDING_MODEL)
    embed_dim = int(os.getenv("EVAL_RAG_EMBED_DIM", str(settings.LOG_EMBEDDING_DIMENSIONS)))

    data = _load_dataset(DATASET_PATH)
    incidents = data["incidents"]
    slug_list = _vocabulary(incidents)

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(int(os.getenv("EVAL_RAG_CONCURRENCY", "3")))

    rows = await asyncio.gather(
        *[_eval_one_incident(sem, client, embed_model, embed_dim, judge_model, slug_list, inc) for inc in incidents]
    )

    def col(key_path: str) -> list[bool]:
        out = []
        for r in rows:
            cur = r
            for part in key_path.split("."):
                cur = cur[part]
            out.append(bool(cur))
        return out

    def col_recall(mode: str) -> list[float]:
        return [float(r["recall_at_5"][mode]) for r in rows]

    agg = {
        "llm_no_retrieval": {
            "root_cause_exact": _mean(col("root_cause_exact.llm_no_retrieval")),
            "root_cause_top3": _mean(col("root_cause_top3.llm_no_retrieval")),
            "recall_at_5": None,
        },
        "naive_fulltext": {
            "root_cause_exact": _mean(col("root_cause_exact.naive_fulltext")),
            "root_cause_top3": _mean(col("root_cause_top3.naive_fulltext")),
            "recall_at_5": _mean_f(col_recall("naive_fulltext")),
        },
        "rag_embedding": {
            "root_cause_exact": _mean(col("root_cause_exact.rag_embedding")),
            "root_cause_top3": _mean(col("root_cause_top3.rag_embedding")),
            "recall_at_5": _mean_f(col_recall("rag_embedding")),
        },
    }

    def pp_delta(a: float, b: float) -> float:
        return round((b - a) * 100.0, 2)

    report = {
        "eval_name": "rag_root_cause",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_doc": "evals/rag_root_cause/METHODOLOGY.md",
        "dataset": str(DATASET_PATH.relative_to(_ROOT)),
        "n_incidents": len(incidents),
        "k_retrieval": K_RETRIEVAL,
        "models": {"judge": judge_model, "embedding": embed_model, "embedding_dim": embed_dim},
        "aggregate_metrics": agg,
        "deltas_percentage_points": {
            "naive_vs_no_retrieval_root_cause_exact_pp": pp_delta(
                agg["llm_no_retrieval"]["root_cause_exact"], agg["naive_fulltext"]["root_cause_exact"]
            ),
            "rag_vs_no_retrieval_root_cause_exact_pp": pp_delta(
                agg["llm_no_retrieval"]["root_cause_exact"], agg["rag_embedding"]["root_cause_exact"]
            ),
            "rag_vs_no_retrieval_root_cause_top3_pp": pp_delta(
                agg["llm_no_retrieval"]["root_cause_top3"], agg["rag_embedding"]["root_cause_top3"]
            ),
            "rag_vs_naive_recall_at_5_mean_pp": pp_delta(
                agg["naive_fulltext"]["recall_at_5"] or 0.0,
                agg["rag_embedding"]["recall_at_5"] or 0.0,
            ),
        },
        "human_rubric": {
            "note": "Template in METHODOLOGY.md; incidents flagged in dataset with human_rubric_sample: true",
            "flagged_incident_ids": [r["incident_id"] for r in rows if r["human_rubric_sample"]],
        },
        "per_incident": rows,
    }
    return report


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(async_main())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"rag_root_cause_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "per_incident"}, indent=2))
    print(f"\nFull report: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
