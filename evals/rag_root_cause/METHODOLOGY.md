# RAG root-cause evaluation methodology

This document defines how numbers in `evals/reports/rag_root_cause_*.json` are produced so README and resume claims can cite **method**, not hype.

## Dataset

- **Source:** `evals/rag_root_cause/dataset.yaml` — **synthetic** incidents (grow toward 50–200 rows).
- **Each incident** includes:
  - `description`: natural-language incident summary (simulates user/engineer input).
  - `gold_root_cause`: slug from a **closed vocabulary** (all slugs appear in the file).
  - `gold_evidence_chunk_ids`: chunk ids that should be retrieved as supporting evidence.
  - `chunks`: mini log corpus mixing evidence (`is_evidence: true`) and distractors.

No production database is required; each incident is self-contained.

## Baselines

| Code | Meaning |
|------|--------|
| **llm_no_retrieval** | Judge model sees **only** the incident description + **closed list of slugs**. No log excerpts. |
| **naive_fulltext** | Chunks ranked by **token overlap** between description and chunk text (case-insensitive, tokens length > 2). Top **K=5** texts concatenated and passed to the **same** judge model. |
| **rag_embedding** | Chunks ranked by **cosine similarity** of OpenAI embeddings (query vs each chunk). Top **K=5** passed to the judge model. |

All three use the **same** judge model (default `gpt-4o-mini`, override `EVAL_RAG_JUDGE_MODEL`) and **JSON** output: `root_cause` + `alternatives` (two other slugs).

## Metrics

### Root cause — exact

- **1** if `root_cause` equals `gold_root_cause` (string match).
- **0** otherwise.

Reported as the **mean** over incidents (equivalent to accuracy on this closed set).

### Root cause — top-3

- **1** if `gold_root_cause` appears in `[root_cause, alternatives[0], alternatives[1]]`.
- **0** otherwise.

Mean over incidents.

### Recall@K on evidence (K = 5)

- Gold set \(G\) = `gold_evidence_chunk_ids`.
- Retrieved set \(T\) = top-5 chunk ids by ranking (naive or embedding).
- **Recall@5** = \(|T \cap G| / |G|\) (if \(|G|=0\), defined as 1.0 for that incident).

Reported as the **mean** over incidents for naive and RAG rankings. **Not** defined for `llm_no_retrieval` (no retrieval).

### “+30%” or any percentage headline

- A claim like **“+30% root-cause accuracy”** is **only** valid if it refers to a **documented delta** on this harness, e.g.:
  - `deltas_percentage_points.rag_vs_no_retrieval_root_cause_exact_pp` in the JSON report  
  - which is **100 × (mean_exact_rag − mean_exact_no_retrieval)** on **this** dataset, **this** judge model, **this** date.
- If the measured delta is not ~30, **do not** round or market it as 30%. Update the resume/README to the **actual** number or say “improved on synthetic eval (see report).”

### Human rubric (subset)

- Incidents with `human_rubric_sample: true` in YAML are candidates for manual review.
- Use a short form (example):
  - **A.** Is gold root cause fair for the description? (Y/N)
  - **B.** Are gold evidence ids the right lines? (Y/N)
  - **C.** Optional 1–5 quality of model explanation (if you log free text later).

Aggregate **human** scores are **not** auto-filled; they require a separate pass and optional second JSON artifact.

## Reproduction

```bash
make eval-rag
```

Or:

```bash
source scripts/set_env.sh development   # loads .env.development
PYTHONPATH=. uv run python -m evals.rag_root_cause.run
```

**Requires:** `OPENAI_API_KEY` (or `EVALUATION_API_KEY`).

**Optional env:**

| Variable | Purpose |
|----------|---------|
| `EVAL_RAG_JUDGE_MODEL` | Default `gpt-4o-mini` |
| `EVAL_RAG_EMBED_MODEL` | Default `LOG_EMBEDDING_MODEL` |
| `EVAL_RAG_EMBED_DIM` | Default `LOG_EMBEDDING_DIMENSIONS` |
| `EVAL_RAG_CONCURRENCY` | Parallel incidents (default `3`) |

## Limitations

- **Synthetic** data; does not prove production lift until repeated on **anonymized real** incidents.
- Judge is **LLM-as-judge**; bias and variance exist—report model name and seed/temperature (temperature fixed at 0.2 in code).
- Closed vocabulary **simplifies** the task vs open-ended RCA in production.

## Artifacts

- **Per run:** `evals/reports/rag_root_cause_<UTC_timestamp>.json`
- **Methodology:** this file (`evals/rag_root_cause/METHODOLOGY.md`)
