# Phase 5 Report — Candidate Generation

**Run:** 2026-10-18T04:07:11.695976+00:00  
**Queries:** 50 golden queries · **Output depth:** top-20 · **Pool depth:** top-50 · **Per-branch candidates:** 20 · **RRF k:** 60  
**Reformulations:** fixed cache (`evaluation/cache/reformulations.json`) · **Vector search:** exact cosine, in-process (bge-small-en-v1.5)

## 1. Overall

| Configuration | R@5 | R@10 | R@20 | MRR | NDCG@5 | NDCG@10 | Pool R@50 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Vector, plain passages | 0.5000 | 0.6400 | 0.6900 | 0.4754 | 0.3756 | 0.4238 | 0.8000 |
| Vector, contextual passages | 0.6000 | 0.7000 | 0.8200 | 0.5644 | 0.4887 | 0.5222 | 0.8800 |
| BM25 (title+section+text) | 0.6400 | 0.7400 | 0.8400 | 0.5822 | 0.5064 | 0.5339 | 0.8900 |
| Hybrid: contextual vec + BM25 (no reformulation) | 0.6400 | 0.7900 | 0.8600 | 0.5868 | 0.5270 | 0.5673 | 0.9100 |
| Pipeline: plain vec + BM25 | 0.6600 | 0.8200 | 0.9200 | 0.6469 | 0.5299 | 0.5829 | 0.9700 |
| Pipeline: contextual vec + BM25 | 0.6900 | 0.8300 | 0.9300 | 0.6654 | 0.5771 | 0.6275 | 0.9800 |
| Pipeline: contextual vec + BM25, candidate_k=50 (Phase 5) | 0.7000 | 0.8700 | 0.9400 | 0.6678 | 0.5888 | 0.6430 | 0.9800 |

## 2. Per-category: Pipeline: plain vec + BM25 → Pipeline: contextual vec + BM25, candidate_k=50 (Phase 5)

| Category | R@10 before | R@10 after | MRR before | MRR after | NDCG@10 before | NDCG@10 after |
|:---|---:|---:|---:|---:|---:|---:|
| `abbreviation` | 0.8000 | 0.8000 | 0.4000 | 0.4900 | 0.4987 | 0.5554 |
| `course_code` | 0.9000 | 1.0000 | 0.8667 | 0.7667 | 0.7420 | 0.7602 |
| `eligibility` | 0.8000 | 0.9000 | 0.7667 | 0.7500 | 0.6683 | 0.7301 |
| `keyword` | 0.9000 | 0.8000 | 0.7200 | 0.7143 | 0.6012 | 0.6245 |
| `lexical_diversity` | 1.0000 | 1.0000 | 0.7667 | 0.5667 | 0.6767 | 0.6300 |
| `natural_language` | 0.7000 | 0.8000 | 0.4619 | 0.6200 | 0.3749 | 0.5572 |
| `numeric` | 0.7000 | 0.8000 | 0.7400 | 0.7667 | 0.5554 | 0.5957 |
| `policy` | 1.0000 | 1.0000 | 0.5833 | 0.7067 | 0.6551 | 0.7352 |
| `procedure` | 0.9000 | 0.9000 | 0.8667 | 0.8286 | 0.7739 | 0.7808 |
| `typo` | 0.5000 | 0.7000 | 0.2975 | 0.4686 | 0.2824 | 0.4605 |

## 3. Significance: Pipeline: contextual vec + BM25, candidate_k=50 (Phase 5) vs Pipeline: plain vec + BM25

Paired randomization test (two-sided) and 95% bootstrap CI of the per-query difference; 10,000 resamples, seed 0.

| Metric | Before | After | Δ | 95% CI | p-value | Significant (p<0.05) |
|:---|---:|---:|---:|:---:|---:|:---:|
| recall@10 | 0.8200 | 0.8700 | +0.0500 | [+0.0000, +0.1000] | 0.1243 | no |
| mrr | 0.6469 | 0.6678 | +0.0209 | [-0.0625, +0.1061] | 0.6361 | no |
| ndcg@10 | 0.5829 | 0.6430 | +0.0601 | [+0.0108, +0.1122] | 0.0258 | yes |
| pool_recall@50 | 0.9700 | 0.9800 | +0.0100 | [+0.0000, +0.0300] | 1.0000 | no |

## 4. Improved queries (13)

| Query | Category | Text | MRR before | MRR after | R@10 before | R@10 after |
|:---|:---|:---|---:|---:|---:|---:|
| Q003 | `keyword` | CSE301 prerequisites | 0.500 | 1.000 | 1.00 | 1.00 |
| Q006 | `natural_language` | Can I get a refund if I withdraw from the semester? | 0.500 | 1.000 | 1.00 | 1.00 |
| Q007 | `natural_language` | What are the requirements for hostel allotment? | 0.167 | 1.000 | 0.50 | 0.50 |
| Q011 | `abbreviation` | sem fees payment schedule | 0.500 | 1.000 | 1.00 | 1.00 |
| Q017 | `typo` | hostel allotmnt process | 0.167 | 1.000 | 0.50 | 1.00 |
| Q020 | `typo` | semester withdrawl procedure | 0.067 | 0.100 | 0.00 | 0.50 |
| Q027 | `numeric` | minimum CGPA 6.0 academic probation | 0.200 | 0.333 | 0.50 | 0.50 |
| Q028 | `numeric` | caution deposit refund within 30 days | 1.000 | 1.000 | 0.50 | 1.00 |
| Q032 | `policy` | zero tolerance anti ragging policy | 0.167 | 0.200 | 1.00 | 1.00 |
| Q033 | `policy` | student grievance redressal ombudsman policy | 0.250 | 0.333 | 1.00 | 1.00 |
| Q034 | `policy` | academic integrity and plagiarism prevention policy | 0.500 | 1.000 | 1.00 | 1.00 |
| Q040 | `eligibility` | internship NOC eligibility and credit transfer | 0.333 | 1.000 | 1.00 | 1.00 |
| Q048 | `lexical_diversity` | LibNet OPAC book reservation and account renewal guide | 0.333 | 0.500 | 1.00 | 1.00 |

## 5. Regressed or mixed queries (12)

| Query | Category | Text | MRR before | MRR after | R@10 before | R@10 after |
|:---|:---|:---|---:|---:|---:|---:|
| Q002 | `keyword` | hostel allotment | 1.000 | 0.500 | 1.00 | 1.00 |
| Q005 | `keyword` | library book renewal | 0.100 | 0.071 | 0.50 | 0.00 |
| Q008 | `natural_language` | What happens if my attendance is below 75%? | 0.143 | 0.100 | 0.50 | 0.50 |
| Q009 | `natural_language` | How many credits can an undergraduate student register per semester? | 1.000 | 0.500 | 0.50 | 1.00 |
| Q015 | `abbreviation` | prof office hours and mentoring guidelines | 0.250 | 0.200 | 1.00 | 1.00 |
| Q016 | `typo` | attendence shortage rules | 0.111 | 0.100 | 0.50 | 0.50 |
| Q021 | `course_code` | cse301 syllabus and textbooks | 1.000 | 0.500 | 0.50 | 1.00 |
| Q038 | `eligibility` | eligibility for supplementary backlog examination | 0.500 | 0.250 | 0.50 | 1.00 |
| Q039 | `eligibility` | eligibility for student semester exchange program | 1.000 | 0.500 | 1.00 | 1.00 |
| Q045 | `procedure` | procedure for on-duty leave application and approval | 0.333 | 0.143 | 0.50 | 0.50 |
| Q047 | `lexical_diversity` | ExamShield portal hall ticket download and re-evaluation filing | 1.000 | 0.333 | 1.00 | 1.00 |
| Q049 | `lexical_diversity` | campus Wi-Fi 802.1X PEAP MSCHAPv2 device authentication | 1.000 | 0.500 | 1.00 | 1.00 |

## 6. Notes

- With 50 queries, one query moves Recall@10 by 0.01–0.02. Treat differences below ~0.03 as noise.
- BM25 indexes title, section and chunk text only; the synthetic `keywords` metadata is deliberately excluded, because it was produced by the same generator as the golden queries.
- Contextual passages change only the embedded text; the stored `chunk_text` is unchanged. Production vector search picks this up after re-running ingestion (`EMBEDDING_CONTEXTUAL=1`, the default).
