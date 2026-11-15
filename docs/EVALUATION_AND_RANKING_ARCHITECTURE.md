# Evaluation Judge vs Production Reranker: Architecture and Dependencies

**Date:** 2026-11-13 · **Base commit:** `e68c9f0`.

## 1. Why there are two components

| | Production reranker | Evaluation judge |
|:---|:---|:---|
| Purpose | Reorder hybrid-retrieval candidates for users | Offline diagnosis of retrieval quality |
| Module | `reformulation/production_reranker.py` | `evaluation/judge/` |
| Model | Cross-encoder (`ms-marco-MiniLM-L-6-v2` by default) | LLM on its own provider key (default `groq` / `openai/gpt-oss-120b`), independent of the reformulation model (`gemini-3.5-flash-lite`) |
| Output | Reordered chunks; raw logit in `metadata["reranker_logit"]` | 0–3 grades with verbatim evidence spans, intents, answerability (relative to the judged passages) |
| Meaning of the score | Uncalibrated ordering score for **one query** | An LLM opinion under a fixed rubric; **not ground truth** |
| Used by | `api/app.py` (when `RERANKER_ENABLED=true`), `jury_demo.py`, `evaluation/run_ad_benchmark.py` | `evaluation/run_ad_benchmark.py --judge-sample N`, `jury_demo.py --judge` |
| Never | Interprets logits as probabilities; applies thresholds; produces relevance labels | Changes rankings; feeds production; writes label files; declares an improvement |

**Why this matters.** The previous demo used the same cross-encoder both to judge relevance (with `logit > 0 = relevant`) and as a candidate reranker. That is circular evaluation, and the audit showed it misleading:
- its improved/reduced verdict agreed with the labels in only 42 of 65 decided held-out queries;
- it scored a misspelled query's correct documents at the model's floor.

The ground truth is human labels (`evaluation/datasets/`).

## 2. Phase 0 inventory and reuse decisions

| Existing component | Location | Decision |
|:---|:---|:---|
| Hybrid retrieval (dense + BM25, RRF k=60) | `retrieval/hybrid_retriever.py`, `bm25_retriever.py`, `vector_retriever.py` | **Reused unchanged** |
| Multi-query RRF | `reformulation/search_pipeline.py::aggregate_retrieval` | **Reused unchanged** |
| Contextual passage builder | `reformulation/cross_encoder.py::build_contextual_passage` | **Reused** by the production reranker |
| Legacy `CrossEncoderReranker` | `reformulation/cross_encoder.py` | Kept for existing evaluation scripts/tests; the API and demo no longer use it. Its `score()` was the old demo "judge" |
| Search pipeline reranking step | `search_pipeline.py::search_detailed` | **Changed**: with a production reranker, retrieval fetches `max(final_k, candidate_limit)` candidates, reranks, then cuts to `final_k`. Previously it reranked only the already-cut list. Legacy path unchanged |
| API | `api/app.py` | **Changed**: `RerankerConfig.from_env()`; `reranker_logit` field; no judge import |
| Demo verdict logic | `jury_demo.py` | **Removed**: logit threshold, mean-logit verdict, cross-encoder judge |
| Metrics, significance | `evaluation/metrics.py`, `evaluation/significance.py` | **Reused** |
| Local exact vector index, cached rewrites | `evaluation/run_phase5_evaluation.py`, `evaluation/cache/` | **Reused** by the A–D runner |
| Query sets and labels | `evaluation/golden_queries.json`, `evaluation/heldout_queries.json` | **Reused**, now with explicit split status in `evaluation/datasets/manifest.json` |

## 3. Dependency rules (enforced by `tests/test_component_separation.py`)

```
                         ┌───────────────────────── production ─────────────────────────┐
  api/app.py ──────────► reformulation/search_pipeline.py ──► retrieval/*
       │                         │
       └────────────────► reformulation/production_reranker.py ──► reformulation/cross_encoder.build_contextual_passage
                                                                     (no import of evaluation/*)
                         └──────────────────────────────────────────────────────────────┘
                                         ▲ may call (read-only)
  evaluation/run_ad_benchmark.py ────────┘
       │
       └──► evaluation/judge/ (rubric, schema, llm_judge) ──► reformulation/llm (HTTP client only)
              (no import of reranker, pipeline, retrieval or api)
```

| Rule | Test |
|:---|:---|
| api/, reformulation/, retrieval/, embedding/ never import `evaluation.*` or anything named "judge" | `test_production_code_never_imports_evaluation_or_judge` |
| api/app.py never references the judge | `test_api_does_not_reference_the_judge` |
| The judge never imports the reranker, pipeline, reformulator, retrieval or api | `test_judge_does_not_import_ranking_or_retrieval` |
| The reranker never imports evaluation code | `test_reranker_does_not_import_evaluation` |
| The reranker exposes no threshold, probability, relevance, grade, label or calibration surface | `test_reranker_has_no_label_probability_or_threshold_surface` |
| The judge's API accepts no scores; its prompt never contains fusion scores, logits or chunk IDs; it does not mutate inputs | `test_judge_api_does_not_accept_reranker_scores`, `test_judge_prompt_never_contains_reranker_or_fusion_scores` |

**Known limitation.** Importing `evaluation.judge` executes `evaluation/__init__.py`, which imports the older evaluator, and that evaluator imports the search pipeline. This is a package-initialisation side effect of the evaluation package, not a dependency of the judge's own code. The rules above are therefore enforced on each module's own imports (AST) and by runtime tests.

## 4. Evaluation judge design

- **Rubric:** `evaluation/judge/rubric.py`. `RUBRIC_VERSION = relevance-rubric-v1` and `PROMPT_VERSION = judge-prompt-v1`; any text change must bump them. Grades run 0–3, with multi-intent support (intents listed once per query, and each passage names the intents it addresses).
- **Schema validation:** `evaluation/judge/schema.py`. The required keys are exact and grades are integers 0–3.
  - Grades 1–3 need 1–5 evidence spans, each a verbatim substring of the passage (whitespace- and case-insensitive).
  - Grade 0 must have no evidence.
  - Grades 2–3 must name the intents they address.
  - Every passage must be judged exactly once.
  - **Invalid output is rejected and never repaired.** It gets one retry, then the result is `status="rejected"` with no grades.
- **Answerability:** derived deterministically from the validated grades. It is "fully" when every intent is covered by a grade ≥ 2 passage, "partially" when some are, and "unanswerable" when none are. It is relative to the judged passages, not the corpus.
- **Blindness:** passages are de-duplicated and ordered by `sha256(query_id, chunk_id)` and shown as P1…Pn. The prompt never contains system names, ranks, scores or chunk IDs.
- **Provenance per result:** query ID, provider, model, rubric and prompt versions, a UTC timestamp, the passage order and the raw responses.

## 5. Production reranker design

- **Config:** `RerankerConfig`: `enabled` (default off), `model_name`, `candidate_limit` (1–100, typically 20/50/100), `batch_size`, `max_passage_chars`, `local_files_only`, `revision`. Environment variables: `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANKER_CANDIDATE_LIMIT`, `RERANKER_BATCH_SIZE`, `RERANKER_REVISION`.
- **Scoring:** contextual `Title / Section / Content` passages only; there is no text-only mode. Results are sorted by descending logit, with ties broken by retrieval order. IDs and the retrieval `score` are kept, and the logit is added to the metadata.
- **Safety:** empty input gives empty output. Any model or inference error keeps the retrieval order, sets `reranked=False`, records the error and marks the search result `degraded`.
- **Latency** is recorded in `RerankOutcome.latency_ms` and the pipeline's `timings_ms["reranking"]`.
- **Status:** **off in production.** See `evaluation/results/AD_BENCHMARK_REPORT.md` for why.
