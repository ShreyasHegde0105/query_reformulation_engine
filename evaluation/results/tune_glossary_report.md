# Held-out Evaluation — `tune_glossary`

**Run:** 2026-10-20T10:41:58.037191+00:00  
**Queries:** 50 answerable (`evaluation/golden_queries.json`)  
**Reformulations:** `evaluation/cache/golden_queries_reformulations_gemini-3.5-flash-lite_glossary.json` · **Output depth:** top-20 · **Pool depth:** top-50 · **Per-branch candidates:** max(20, depth) · **RRF k:** 60

## 1. Overall

| System | R@5 | R@10 | R@20 | MRR | NDCG@10 | Pool R@50 |
|:---|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.6400 | 0.7400 | 0.8400 | 0.5822 | 0.5339 | 0.8900 |
| Vector only (contextual passages) | 0.6000 | 0.7000 | 0.8200 | 0.5644 | 0.5222 | 0.8800 |
| Hybrid, raw query (no query processing) | 0.6400 | 0.7900 | 0.8600 | 0.5868 | 0.5673 | 0.9100 |
| Pipeline: normalization + spelling, no LLM | 0.6500 | 0.7900 | 0.8800 | 0.5816 | 0.5633 | 0.9300 |
| Pipeline: normalization + LLM rewrites, no spelling | 0.6100 | 0.7800 | 0.8700 | 0.5740 | 0.5615 | 0.9800 |
| Full pipeline: normalization + spelling + LLM rewrites | 0.6100 | 0.7800 | 0.8800 | 0.5756 | 0.5615 | 0.9800 |

## 2. What each layer adds

Paired randomization test (two-sided), 95% bootstrap CI, 10,000 resamples, seed 0. Holm-adjusted p corrects for all 9 tests; "Significant" uses the adjusted p < 0.05.

| Layer | Metric | Before | After | Δ | 95% CI | p | Holm p | Significant |
|:---|:---|---:|---:|---:|:---:|---:|---:|:---:|
| Normalization + spelling | recall@10 | 0.7900 | 0.7900 | +0.0000 | [-0.0300, +0.0300] | 1.0000 | 1.0000 | no |
| Normalization + spelling | mrr | 0.5868 | 0.5816 | -0.0053 | [-0.0451, +0.0368] | 0.8697 | 1.0000 | no |
| Normalization + spelling | ndcg@10 | 0.5673 | 0.5633 | -0.0040 | [-0.0263, +0.0214] | 0.8716 | 1.0000 | no |
| LLM rewrites (on top of spelling) | recall@10 | 0.7900 | 0.7800 | -0.0100 | [-0.0800, +0.0600] | 1.0000 | 1.0000 | no |
| LLM rewrites (on top of spelling) | mrr | 0.5816 | 0.5756 | -0.0060 | [-0.0919, +0.0781] | 0.8882 | 1.0000 | no |
| LLM rewrites (on top of spelling) | ndcg@10 | 0.5633 | 0.5615 | -0.0018 | [-0.0613, +0.0567] | 0.9523 | 1.0000 | no |
| Whole query layer | recall@10 | 0.7900 | 0.7800 | -0.0100 | [-0.0800, +0.0600] | 1.0000 | 1.0000 | no |
| Whole query layer | mrr | 0.5868 | 0.5756 | -0.0113 | [-0.0974, +0.0736] | 0.8024 | 1.0000 | no |
| Whole query layer | ndcg@10 | 0.5673 | 0.5615 | -0.0058 | [-0.0670, +0.0540] | 0.8584 | 1.0000 | no |

## 3. Per category: hybrid (raw query) → full pipeline

| Category | n | R@10 before | R@10 after | MRR before | MRR after | NDCG@10 before | NDCG@10 after |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `abbreviation` | 5 | 0.8000 | 0.8000 | 0.4000 | 0.5000 | 0.4865 | 0.5443 |
| `course_code` | 5 | 0.9000 | 0.7000 | 0.8000 | 0.7154 | 0.7328 | 0.6661 |
| `eligibility` | 5 | 0.9000 | 1.0000 | 0.6400 | 0.6222 | 0.6385 | 0.6669 |
| `keyword` | 5 | 0.7000 | 0.5000 | 0.4917 | 0.5143 | 0.4217 | 0.4058 |
| `lexical_diversity` | 5 | 1.0000 | 1.0000 | 0.6667 | 0.6667 | 0.7062 | 0.6864 |
| `natural_language` | 5 | 0.7000 | 0.8000 | 0.3500 | 0.3767 | 0.3532 | 0.4455 |
| `numeric` | 5 | 0.7000 | 0.6000 | 0.6000 | 0.4772 | 0.4558 | 0.4170 |
| `policy` | 5 | 1.0000 | 1.0000 | 0.8000 | 0.7000 | 0.8313 | 0.6761 |
| `procedure` | 5 | 0.9000 | 0.9000 | 0.7200 | 0.7286 | 0.7227 | 0.7471 |
| `typo` | 5 | 0.3000 | 0.5000 | 0.4000 | 0.4546 | 0.3242 | 0.3598 |

## 4. Per-query effect of the LLM rewrites (NDCG@10, full vs no-LLM pipeline)

Better: 21 · Worse: 20 · Unchanged: 9

| Query | Category | Text | Δ NDCG@10 |
|:---|:---|:---|---:|
| Q040 | `eligibility` | internship NOC eligibility and credit transfer | -0.621 |
| Q035 | `policy` | hostel electrical appliances prohibition policy | -0.525 |
| Q024 | `course_code` | CSE301 practical laboratory assignments | -0.497 |
| Q004 | `keyword` | academic transcript | -0.346 |
| Q029 | `numeric` | maximum 28 credit load per semester | -0.337 |
| Q032 | `policy` | zero tolerance anti ragging policy | -0.295 |
| Q050 | `lexical_diversity` | CareerHub portal CDC registration and one-offer placement norms | -0.242 |
| Q021 | `course_code` | cse301 syllabus and textbooks | -0.169 |
| Q003 | `keyword` | CSE301 prerequisites | -0.169 |
| Q033 | `policy` | student grievance redressal ombudsman policy | -0.125 |
| Q046 | `lexical_diversity` | institutional five-department no-dues clearance sign-off matrix | -0.103 |
| Q006 | `natural_language` | Can I get a refund if I withdraw from the semester? | -0.079 |
| Q008 | `natural_language` | What happens if my attendance is below 75%? | -0.071 |
| Q028 | `numeric` | caution deposit refund within 30 days | -0.070 |
| Q012 | `abbreviation` | dept regulations for lab safety | -0.063 |

## 5. Notes

- One query moves a mean by 1/n; with ~100 queries treat differences below ~0.02 as noise.
- Only the query side differs between systems; indexes, depths and fusion are identical.
- MRR is computed over the top-20 output (MRR@20).
