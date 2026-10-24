# Held-out Evaluation — `tune_set`

**Run:** 2026-10-24T02:32:49.415409+00:00  
**Queries:** 50 answerable (`evaluation/golden_queries.json`)  
**Reformulations:** `evaluation/cache/reformulations.json` · **Output depth:** top-20 · **Pool depth:** top-50 · **Per-branch candidates:** max(20, depth) · **RRF k:** 60

## 1. Overall

| System | R@5 | R@10 | R@20 | MRR | NDCG@10 | Pool R@50 |
|:---|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.6400 | 0.7400 | 0.8400 | 0.5822 | 0.5335 | 0.8900 |
| Vector only (contextual passages) | 0.5900 | 0.7000 | 0.8200 | 0.5634 | 0.5215 | 0.8800 |
| Hybrid, raw query (no query processing) | 0.6400 | 0.7900 | 0.8600 | 0.5868 | 0.5674 | 0.9100 |
| Pipeline: normalization + spelling, no LLM | 0.6500 | 0.7900 | 0.8800 | 0.5816 | 0.5633 | 0.9300 |
| Pipeline: normalization + LLM rewrites, no spelling | 0.7000 | 0.8600 | 0.9400 | 0.6560 | 0.6355 | 0.9800 |
| Full pipeline: normalization + spelling + LLM rewrites | 0.7000 | 0.8600 | 0.9400 | 0.6586 | 0.6383 | 0.9700 |

## 2. What each layer adds

Paired randomization test (two-sided), 95% bootstrap CI, 10,000 resamples, seed 0. Holm-adjusted p corrects for all 9 tests; "Significant" uses the adjusted p < 0.05.

| Layer | Metric | Before | After | Δ | 95% CI | p | Holm p | Significant |
|:---|:---|---:|---:|---:|:---:|---:|---:|:---:|
| Normalization + spelling | recall@10 | 0.7900 | 0.7900 | +0.0000 | [-0.0300, +0.0300] | 1.0000 | 1.0000 | no |
| Normalization + spelling | mrr | 0.5868 | 0.5816 | -0.0053 | [-0.0451, +0.0368] | 0.8697 | 1.0000 | no |
| Normalization + spelling | ndcg@10 | 0.5674 | 0.5633 | -0.0041 | [-0.0264, +0.0214] | 0.8716 | 1.0000 | no |
| LLM rewrites (on top of spelling) | recall@10 | 0.7900 | 0.8600 | +0.0700 | [+0.0100, +0.1300] | 0.0652 | 0.4564 | no |
| LLM rewrites (on top of spelling) | mrr | 0.5816 | 0.6586 | +0.0771 | [-0.0069, +0.1631] | 0.0848 | 0.4564 | no |
| LLM rewrites (on top of spelling) | ndcg@10 | 0.5633 | 0.6383 | +0.0750 | [+0.0184, +0.1341] | 0.0161 | 0.1449 | no |
| Whole query layer | recall@10 | 0.7900 | 0.8600 | +0.0700 | [+0.0100, +0.1300] | 0.0660 | 0.4564 | no |
| Whole query layer | mrr | 0.5868 | 0.6586 | +0.0718 | [-0.0159, +0.1609] | 0.1166 | 0.4664 | no |
| Whole query layer | ndcg@10 | 0.5674 | 0.6383 | +0.0709 | [+0.0124, +0.1319] | 0.0251 | 0.2008 | no |

## 3. Per category: hybrid (raw query) → full pipeline

| Category | n | R@10 before | R@10 after | MRR before | MRR after | NDCG@10 before | NDCG@10 after |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `abbreviation` | 5 | 0.8000 | 0.8000 | 0.4000 | 0.4900 | 0.4865 | 0.5554 |
| `course_code` | 5 | 0.9000 | 1.0000 | 0.8000 | 0.7667 | 0.7328 | 0.7602 |
| `eligibility` | 5 | 0.9000 | 0.9000 | 0.6400 | 0.7500 | 0.6393 | 0.7301 |
| `keyword` | 5 | 0.7000 | 0.8000 | 0.4917 | 0.7143 | 0.4217 | 0.6245 |
| `lexical_diversity` | 5 | 1.0000 | 1.0000 | 0.6667 | 0.5667 | 0.7062 | 0.6300 |
| `natural_language` | 5 | 0.7000 | 0.7000 | 0.3500 | 0.6182 | 0.3532 | 0.5377 |
| `numeric` | 5 | 0.7000 | 0.8000 | 0.6000 | 0.6500 | 0.4558 | 0.5489 |
| `policy` | 5 | 1.0000 | 1.0000 | 0.8000 | 0.7067 | 0.8313 | 0.7264 |
| `procedure` | 5 | 0.9000 | 0.9000 | 0.7200 | 0.8286 | 0.7227 | 0.7808 |
| `typo` | 5 | 0.3000 | 0.7000 | 0.4000 | 0.4952 | 0.3242 | 0.4891 |

## 4. Per-query effect of the LLM rewrites (NDCG@10, full vs no-LLM pipeline)

Better: 26 · Worse: 11 · Unchanged: 13

| Query | Category | Text | Δ NDCG@10 |
|:---|:---|:---|---:|
| Q032 | `policy` | zero tolerance anti ragging policy | -0.483 |
| Q050 | `lexical_diversity` | CareerHub portal CDC registration and one-offer placement norms | -0.414 |
| Q033 | `policy` | student grievance redressal ombudsman policy | -0.211 |
| Q026 | `numeric` | attendance below 75 percent medical condonation | -0.179 |
| Q008 | `natural_language` | What happens if my attendance is below 75%? | -0.169 |
| Q021 | `course_code` | cse301 syllabus and textbooks | -0.056 |
| Q003 | `keyword` | CSE301 prerequisites | -0.049 |
| Q040 | `eligibility` | internship NOC eligibility and credit transfer | -0.049 |
| Q009 | `natural_language` | How many credits can an undergraduate student register per semester? | -0.035 |
| Q012 | `abbreviation` | dept regulations for lab safety | -0.011 |
| Q030 | `numeric` | late fee fine 500 rupees per week | -0.008 |

## 5. Notes

- One query moves a mean by 1/n; with ~100 queries treat differences below ~0.02 as noise.
- Only the query side differs between systems; indexes, depths and fusion are identical.
- MRR is computed over the top-20 output (MRR@20).
