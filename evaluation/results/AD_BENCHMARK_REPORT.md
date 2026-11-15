# A–D Benchmark Report: Reformulation × Production Reranker

**Date:** 2026-11-13 · **Code:** commit `e68c9f0` plus uncommitted working-tree changes (judge, production reranker, runner) · **Runner:** `evaluation/run_ad_benchmark.py`
**Raw results:** `evaluation/results/ad_benchmark_development_c50_ms-marco-MiniLM-L-6-v2.{json,md}`

Every number here was measured. Anything not measured is marked **NOT EVALUATED**, with the reason.

## 1. What was run

| Run | Split | Candidate limit | Reranker | Status |
|:---|:---|---:|:---|:---|
| 1 | development (50 queries) | 50 | ms-marco-MiniLM-L-6-v2 | **Completed** |
| 2 | validation_contaminated (94 labelled) + LLM judge on 20 sampled queries | 50 | ms-marco-MiniLM-L-6-v2 | **NOT EVALUATED**: the background suite was stopped by Claude Code because the machine ran critically low on memory (0.9 GB free of 7.8 GB) |
| 3–4 | development / validation | 20 | ms-marco-MiniLM-L-6-v2 | **NOT EVALUATED**: same stop |
| 5 | development | 100 | ms-marco-MiniLM-L-6-v2 | **NOT EVALUATED**: same stop |
| 6–7 | development / validation | 20 | BAAI/bge-reranker-base | **NOT EVALUATED**: same stop |
| n/a | blind_test (66 queries) | n/a | n/a | **NOT EVALUATED**: no human labels; the split refuses to load until labels are frozen |

To finish, run `bash audit/run_ad_suite.sh` (sequential; ≈2 h on this CPU) when memory allows.

## 2. Configuration (Run 1)

- **Corpus:** `Data/chunks/chunks.jsonl`, 702 chunks, sha256 `695ee680f1fc7cea…`.
- **Queries:** `evaluation/golden_queries.json` (development / tuning set), sha256 `0bbac716…`. Label provenance is unrecorded.
- **Rewrites:** `evaluation/cache/reformulations.json`. The rewrite model and prompt are not recorded in that cache, so the prompt is assumed to be `generic-v1`.
- **Retrieval:** hybrid (bge-small-en-v1.5 exact cosine on contextual passages + BM25), RRF k = 60. Branch depth 50, candidate pool 50, final k 10, ≤ 5 rewrites. Routing is **off** in all four configurations; the corpus spell corrector is **on** in all four.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, contextual passages, raw logits, no threshold.
- **Differences between A–D:** only rewrites on/off and the reranker on/off.

## 3. Results: development set (n = 50, labels provenance unrecorded)

| Config | P@5 | P@10 | R@5 | R@10 | MRR@10 | NDCG@10 | Hit@10 | Unique docs@10 | Dup-doc rate | Retrieval ms mean/p95 | Rerank ms mean/p95 | End-to-end ms excl. LLM mean/p95 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** original | 0.260 | 0.154 | 0.650 | 0.770 | 0.573 | 0.556 | 0.840 | 5.76 | 0.424 | 4 / 5 | n/a | 46 / 54 |
| **B** reformulated | 0.276 | 0.172 | 0.690 | 0.860 | 0.652 | 0.624 | 0.960 | 5.42 | 0.458 | 17 / 23 | n/a | 229 / 264 |
| **C** original + reranker | 0.268 | 0.158 | 0.670 | 0.790 | 0.663 | 0.596 | 0.880 | 5.48 | 0.452 | 4 / 6 | 5,507 / 7,575 | 5,512 / 7,580 |
| **D** reformulated + reranker | 0.268 | 0.158 | 0.670 | 0.790 | 0.663 | 0.596 | 0.880 | 5.54 | 0.446 | 19 / 26 | 5,316 / 6,607 | 5,337 / 6,628 |

**Latency notes**
- The figures are in-process on CPU.
- The **live LLM rewrite call is not included**, because the runs used cached rewrites. A separate live measurement (audit, n = 8) found about 1.1 s per call.
- The B/D "end-to-end" time includes embedding the rewrites for the redundancy filter.

### Changes relative to the appropriate baseline

Paired randomization test with a 95% bootstrap CI; Holm correction across all 20 rows of the raw report.

| Comparison | Metric | Baseline → system | Abs. Δ | Rel. Δ | 95% CI | p | Holm p |
|:---|:---|:---|---:|---:|:---:|---:|---:|
| A → B (reformulation) | R@10 | 0.770 → 0.860 | +0.090 | +11.7% | [+0.020, +0.160] | 0.036 | 0.64 |
| | MRR@10 | 0.573 → 0.652 | +0.080 | +13.9% | [−0.011, +0.172] | 0.098 | 1.00 |
| | NDCG@10 | 0.556 → 0.624 | +0.068 | +12.2% | [+0.005, +0.134] | 0.051 | 0.82 |
| A → C (reranker) | R@10 | 0.770 → 0.790 | +0.020 | +2.6% | [−0.030, +0.070] | 0.68 | 1.00 |
| | MRR@10 | 0.573 → 0.663 | +0.090 | +15.7% | [+0.028, +0.156] | 0.008 | 0.16 |
| | NDCG@10 | 0.556 → 0.596 | +0.040 | +7.2% | [−0.000, +0.083] | 0.069 | 1.00 |
| C → D (reformulation given reranker) | all | identical | 0.000 | 0.0% | [0, 0] | 1.00 | 1.00 |
| B → D (reranker given reformulation) | R@10 | 0.860 → 0.790 | −0.070 | −8.1% | [−0.140, +0.000] | 0.093 | 1.00 |
| | NDCG@10 | 0.624 → 0.596 | −0.028 | −4.4% | [−0.104, +0.047] | 0.50 | 1.00 |

## 4. Reading the results (development set only)

1. **No difference is significant after Holm correction.** With 50 queries, the run cannot establish an improvement for any configuration.
2. **Reformulation without reranking (B)** has the best R@10 and NDCG@10. Its uncorrected confidence intervals exclude zero for R@10 and NDCG@10, and that direction matches earlier runs on both query sets (first audit).
3. **The reranker (C)** raises MRR@10 (+0.090, uncorrected p = 0.008) but adds about **5.5 s per query** on CPU. Recall@10 is unchanged.
4. **C and D are identical on all four metrics, and their top-10 lists are identical for 41 of 50 queries.** This is not a bug: D used its rewrites (6 query variants vs 1). The reranker scores every candidate against the *original* query only, so once it is on, rewrites can matter only by changing which chunks enter the 50-chunk candidate pool. On this corpus the two pools mostly contain the same high-scoring chunks. **With this reranker at a 50-candidate pool, reformulation's ranking benefit is largely removed** (B → D: R@10 −0.070, uncorrected p = 0.09). The 20-candidate runs, which would test whether a smaller pool keeps more of the rewrites' effect, are NOT EVALUATED.
5. **Recommendation:** keep the production reranker **off** (`RERANKER_ENABLED=false`). There is no repeatable, significant benefit; it costs about 5 s per query on CPU; and it cancels most of reformulation's recall gain on this split.

## 5. LLM judge and agreement

| Item | Status |
|:---|:---|
| Judge implementation | Completed and tested (20 unit tests; one live call: Groq `openai/gpt-oss-120b` graded the correct H063 chunk 3 with a verbatim evidence quote, and two non-answers 0) |
| Judge vs existing labels (20-query sample) | **NOT EVALUATED**: part of Run 2, stopped for memory |
| Judge vs human annotations | **NOT EVALUATED**: no labels with recorded human provenance exist |
| Disagreement examples | **NOT EVALUATED** |

## 6. Blind test

- 66 queries are written (34 proposed-unanswerable, 10 typo-heavy, 10 multi-intent, 12 precise), with rewrites cached.
- **No labels**, and the human annotation pool CSV has not been generated (that job was also stopped for memory). Run `python evaluation/datasets/build_annotation_pool.py` when memory allows.
- Every blind-test metric is **NOT EVALUATED**.

## 7. Limitations

- One split (the tuning set), one candidate limit, one reranker model.
- The validation set is development-contaminated, and the blind test is unlabelled.
- Label provenance is unrecorded for both existing query sets.
- Rewrites come from a single cached LLM sample; their variance is not measured.
- Latency was measured on one CPU machine under concurrent load from other jobs, and excludes the LLM call.
