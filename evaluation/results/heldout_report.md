# Held-out Evaluation — `heldout`

**Run:** 2026-10-24T02:30:39.075903+00:00  
**Queries:** 94 answerable (`evaluation/heldout_queries.json`; 6 unanswerable excluded)  
**Reformulations:** `evaluation/cache/heldout_queries_reformulations_gemini-3.5-flash-lite.json` · **Output depth:** top-20 · **Pool depth:** top-50 · **Per-branch candidates:** max(20, depth) · **RRF k:** 60

## 1. Overall

| System | R@5 | R@10 | R@20 | MRR | NDCG@10 | Pool R@50 |
|:---|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.6534 | 0.7917 | 0.8644 | 0.6088 | 0.6106 | 0.9238 |
| Vector only (contextual passages) | 0.7796 | 0.8567 | 0.9277 | 0.7514 | 0.7065 | 0.9730 |
| Hybrid, raw query (no query processing) | 0.7704 | 0.8754 | 0.9330 | 0.6910 | 0.6911 | 0.9775 |
| Pipeline: normalization + spelling, no LLM | 0.7872 | 0.8931 | 0.9454 | 0.6784 | 0.6956 | 0.9775 |
| Pipeline: normalization + LLM rewrites, no spelling | 0.8363 | 0.9374 | 0.9652 | 0.7379 | 0.7429 | 0.9895 |
| Full pipeline: normalization + spelling + LLM rewrites | 0.8452 | 0.9436 | 0.9652 | 0.7271 | 0.7394 | 0.9895 |

## 2. What each layer adds

Paired randomization test (two-sided), 95% bootstrap CI, 10,000 resamples, seed 0. Holm-adjusted p corrects for all 9 tests; "Significant" uses the adjusted p < 0.05.

| Layer | Metric | Before | After | Δ | 95% CI | p | Holm p | Significant |
|:---|:---|---:|---:|---:|:---:|---:|---:|:---:|
| Normalization + spelling | recall@10 | 0.8754 | 0.8931 | +0.0177 | [+0.0000, +0.0408] | 0.1862 | 0.7447 | no |
| Normalization + spelling | mrr | 0.6910 | 0.6784 | -0.0126 | [-0.0435, +0.0170] | 0.4412 | 0.8823 | no |
| Normalization + spelling | ndcg@10 | 0.6911 | 0.6956 | +0.0046 | [-0.0076, +0.0189] | 0.5490 | 0.8823 | no |
| LLM rewrites (on top of spelling) | recall@10 | 0.8931 | 0.9436 | +0.0505 | [+0.0204, +0.0842] | 0.0017 | 0.0136 | yes |
| LLM rewrites (on top of spelling) | mrr | 0.6784 | 0.7271 | +0.0487 | [-0.0018, +0.0999] | 0.0651 | 0.3255 | no |
| LLM rewrites (on top of spelling) | ndcg@10 | 0.6956 | 0.7394 | +0.0438 | [+0.0115, +0.0773] | 0.0089 | 0.0534 | no |
| Whole query layer | recall@10 | 0.8754 | 0.9436 | +0.0683 | [+0.0310, +0.1090] | 0.0003 | 0.0027 | yes |
| Whole query layer | mrr | 0.6910 | 0.7271 | +0.0361 | [-0.0177, +0.0910] | 0.1983 | 0.7447 | no |
| Whole query layer | ndcg@10 | 0.6911 | 0.7394 | +0.0483 | [+0.0146, +0.0827] | 0.0051 | 0.0357 | yes |

## 3. Per category: hybrid (raw query) → full pipeline

| Category | n | R@10 before | R@10 after | MRR before | MRR after | NDCG@10 before | NDCG@10 after |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `abbreviation` | 10 | 0.8583 | 0.9417 | 0.5667 | 0.6000 | 0.6524 | 0.7310 |
| `course_code` | 9 | 1.0000 | 0.9444 | 0.6759 | 0.5778 | 0.7445 | 0.6747 |
| `eligibility` | 9 | 0.9630 | 1.0000 | 0.6481 | 0.6667 | 0.6935 | 0.7103 |
| `keyword` | 9 | 1.0000 | 1.0000 | 0.7593 | 0.8148 | 0.8136 | 0.8289 |
| `lexical_diversity` | 10 | 0.5617 | 0.8367 | 0.4491 | 0.6200 | 0.4295 | 0.6214 |
| `natural_language` | 10 | 0.8167 | 0.8500 | 0.5700 | 0.8500 | 0.5914 | 0.7486 |
| `numeric` | 8 | 0.8333 | 0.9271 | 0.9375 | 1.0000 | 0.7511 | 0.8399 |
| `policy` | 10 | 0.9500 | 0.9500 | 0.6250 | 0.5783 | 0.7246 | 0.6933 |
| `procedure` | 9 | 0.8889 | 1.0000 | 0.7333 | 0.8333 | 0.7125 | 0.7676 |
| `typo` | 10 | 0.9083 | 1.0000 | 1.0000 | 0.7833 | 0.8297 | 0.8010 |

## 4. Per-query effect of the LLM rewrites (NDCG@10, full vs no-LLM pipeline)

Better: 52 · Worse: 32 · Unchanged: 10

| Query | Category | Text | Δ NDCG@10 |
|:---|:---|:---|---:|
| H063 | `policy` | rules about silence hours in hostel | -0.471 |
| H046 | `course_code` | MEC204 heat transfer experiments | -0.318 |
| H062 | `policy` | policy on air conditioner usage in labs and offices | -0.295 |
| H032 | `typo` | intellectual proprety patent filing | -0.217 |
| H008 | `keyword` | exam malpractice penalty | -0.210 |
| H040 | `typo` | librery reading room 24 hours | -0.198 |
| H045 | `course_code` | ECE302 vlsi EDA tools | -0.189 |
| H100 | `lexical_diversity` | websites blocked on campus internet | -0.173 |
| H049 | `course_code` | CSE305 agile project deliverables | -0.121 |
| H076 | `eligibility` | eligibility for research assistant fellowship | -0.110 |
| H084 | `procedure` | how to submit assignment on learnsphere | -0.074 |
| H036 | `typo` | hostle curfew timings | -0.070 |
| H088 | `procedure` | how to request repair of leaking tap in hostel room | -0.056 |
| H052 | `numeric` | late fee after 15 days of registration | -0.048 |
| H077 | `eligibility` | can first year students get a hostel room | -0.038 |

## 5. Notes

- One query moves a mean by 1/n; with ~100 queries treat differences below ~0.02 as noise.
- Only the query side differs between systems; indexes, depths and fusion are identical.
- MRR is computed over the top-20 output (MRR@20).
