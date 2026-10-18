# Held-out Evaluation — `tune_set`

**Run:** 2026-10-18T04:41:21.159423+00:00  
**Queries:** 50 answerable (`evaluation/golden_queries.json`)  
**Reformulations:** `evaluation/cache/reformulations.json` · **Output depth:** top-20 · **Pool depth:** top-50 · **Per-branch candidates:** max(20, depth) · **RRF k:** 60

## 1. Overall

| System | R@5 | R@10 | R@20 | MRR | NDCG@10 | Pool R@50 |
|:---|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.6400 | 0.7400 | 0.8400 | 0.5822 | 0.5339 | 0.8900 |
| Vector only (contextual passages) | 0.6000 | 0.7000 | 0.8200 | 0.5644 | 0.5222 | 0.8800 |
| Hybrid, raw query (no query processing) | 0.6400 | 0.7900 | 0.8600 | 0.5868 | 0.5673 | 0.9100 |
| Pipeline: normalization + spelling, no LLM | 0.6500 | 0.7900 | 0.8800 | 0.5816 | 0.5633 | 0.9300 |
| Pipeline: normalization + LLM rewrites, no spelling | 0.7000 | 0.8700 | 0.9400 | 0.6678 | 0.6430 | 0.9800 |
| Full pipeline: normalization + spelling + LLM rewrites | 0.7000 | 0.8700 | 0.9400 | 0.6700 | 0.6457 | 0.9700 |

## 2. What each layer adds

Paired randomization test (two-sided), 95% bootstrap CI, 10,000 resamples, seed 0. Holm-adjusted p corrects for all 9 tests; "Significant" uses the adjusted p < 0.05.

| Layer | Metric | Before | After | Δ | 95% CI | p | Holm p | Significant |
|:---|:---|---:|---:|---:|:---:|---:|---:|:---:|
| Normalization + spelling | recall@10 | 0.7900 | 0.7900 | +0.0000 | [-0.0300, +0.0300] | 1.0000 | 1.0000 | no |
| Normalization + spelling | mrr | 0.5868 | 0.5816 | -0.0053 | [-0.0451, +0.0368] | 0.8697 | 1.0000 | no |
| Normalization + spelling | ndcg@10 | 0.5673 | 0.5633 | -0.0040 | [-0.0263, +0.0214] | 0.8716 | 1.0000 | no |
| LLM rewrites (on top of spelling) | recall@10 | 0.7900 | 0.8700 | +0.0800 | [+0.0200, +0.1400] | 0.0220 | 0.1533 | no |
| LLM rewrites (on top of spelling) | mrr | 0.5816 | 0.6700 | +0.0884 | [+0.0094, +0.1724] | 0.0424 | 0.2120 | no |
| LLM rewrites (on top of spelling) | ndcg@10 | 0.5633 | 0.6457 | +0.0823 | [+0.0266, +0.1404] | 0.0070 | 0.0630 | no |
| Whole query layer | recall@10 | 0.7900 | 0.8700 | +0.0800 | [+0.0200, +0.1400] | 0.0219 | 0.1533 | no |
| Whole query layer | mrr | 0.5868 | 0.6700 | +0.0832 | [+0.0008, +0.1699] | 0.0631 | 0.2524 | no |
| Whole query layer | ndcg@10 | 0.5673 | 0.6457 | +0.0784 | [+0.0209, +0.1385] | 0.0119 | 0.0952 | no |

## 3. Per category: hybrid (raw query) → full pipeline

| Category | n | R@10 before | R@10 after | MRR before | MRR after | NDCG@10 before | NDCG@10 after |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `abbreviation` | 5 | 0.8000 | 0.8000 | 0.4000 | 0.4900 | 0.4865 | 0.5554 |
| `course_code` | 5 | 0.9000 | 1.0000 | 0.8000 | 0.7667 | 0.7328 | 0.7602 |
| `eligibility` | 5 | 0.9000 | 0.9000 | 0.6400 | 0.7500 | 0.6385 | 0.7301 |
| `keyword` | 5 | 0.7000 | 0.8000 | 0.4917 | 0.7143 | 0.4217 | 0.6245 |
| `lexical_diversity` | 5 | 1.0000 | 1.0000 | 0.6667 | 0.5667 | 0.7062 | 0.6300 |
| `natural_language` | 5 | 0.7000 | 0.8000 | 0.3500 | 0.6200 | 0.3532 | 0.5572 |
| `numeric` | 5 | 0.7000 | 0.8000 | 0.6000 | 0.7667 | 0.4558 | 0.5957 |
| `policy` | 5 | 1.0000 | 1.0000 | 0.8000 | 0.7067 | 0.8313 | 0.7352 |
| `procedure` | 5 | 0.9000 | 0.9000 | 0.7200 | 0.8286 | 0.7227 | 0.7808 |
| `typo` | 5 | 0.3000 | 0.7000 | 0.4000 | 0.4905 | 0.3242 | 0.4875 |

## 4. Per-query effect of the LLM rewrites (NDCG@10, full vs no-LLM pipeline)

Better: 27 · Worse: 10 · Unchanged: 13

| Query | Category | Text | Δ NDCG@10 |
|:---|:---|:---|---:|
| Q032 | `policy` | zero tolerance anti ragging policy | -0.483 |
| Q050 | `lexical_diversity` | CareerHub portal CDC registration and one-offer placement norms | -0.414 |
| Q033 | `policy` | student grievance redressal ombudsman policy | -0.211 |
| Q008 | `natural_language` | What happens if my attendance is below 75%? | -0.071 |
| Q021 | `course_code` | cse301 syllabus and textbooks | -0.056 |
| Q003 | `keyword` | CSE301 prerequisites | -0.049 |
| Q040 | `eligibility` | internship NOC eligibility and credit transfer | -0.049 |
| Q009 | `natural_language` | How many credits can an undergraduate student register per semester? | -0.035 |
| Q012 | `abbreviation` | dept regulations for lab safety | -0.019 |
| Q030 | `numeric` | late fee fine 500 rupees per week | -0.008 |

## 5. Notes

- One query moves a mean by 1/n; with ~100 queries treat differences below ~0.02 as noise.
- Only the query side differs between systems; indexes, depths and fusion are identical.
- MRR is computed over the top-20 output (MRR@20).
