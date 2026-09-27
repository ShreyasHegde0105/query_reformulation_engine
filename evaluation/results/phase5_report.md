# Phase 5 Report — Candidate Generation

**Run:** 2026-09-27T09:40:42.624395+00:00  
**Queries:** 50 golden queries · **Output depth:** top-20 · **Pool depth:** top-50 · **Per-branch candidates:** 20 · **RRF k:** 60  
**Reformulations:** fixed cache (`evaluation/cache/reformulations.json`) · **Vector search:** exact cosine, in-process (bge-small-en-v1.5)

## 1. Overall

| Configuration | R@5 | R@10 | R@20 | MRR | NDCG@5 | NDCG@10 | Pool R@50 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Postgres FTS (Phase 2 keyword) | 0.1000 | 0.1000 | 0.1000 | 0.1540 | 0.1000 | 0.1000 | 0.1000 |
| Phase 2 pipeline: plain vec + Postgres FTS | 0.5400 | 0.6900 | 0.7900 | 0.5615 | 0.4234 | 0.4726 | 0.8700 |
| Vector, plain passages | 0.5000 | 0.6400 | 0.6900 | 0.4754 | 0.3756 | 0.4238 | 0.8000 |
| Vector, contextual passages | 0.6000 | 0.7000 | 0.8200 | 0.5644 | 0.4887 | 0.5222 | 0.8800 |
| BM25 (title+section+text) | 0.6400 | 0.7400 | 0.8400 | 0.5822 | 0.5064 | 0.5339 | 0.8900 |
| Hybrid: contextual vec + BM25 (no reformulation) | 0.6400 | 0.7900 | 0.8600 | 0.5868 | 0.5270 | 0.5673 | 0.8700 |
| Pipeline: plain vec + BM25 | 0.6600 | 0.8200 | 0.9300 | 0.6495 | 0.5288 | 0.5859 | 0.9500 |
| Pipeline: contextual vec + BM25 | 0.6900 | 0.8300 | 0.9300 | 0.6658 | 0.5780 | 0.6285 | 0.9600 |
| Pipeline: contextual vec + BM25, candidate_k=50 (Phase 5) | 0.7000 | 0.8700 | 0.9400 | 0.6696 | 0.5898 | 0.6446 | 0.9600 |

## 2. Per-category: Phase 2 pipeline: plain vec + Postgres FTS → Pipeline: contextual vec + BM25, candidate_k=50 (Phase 5)

| Category | R@10 before | R@10 after | MRR before | MRR after | NDCG@10 before | NDCG@10 after |
|:---|---:|---:|---:|---:|---:|---:|
| `abbreviation` | 0.7000 | 0.8000 | 0.3733 | 0.4900 | 0.4446 | 0.5554 |
| `course_code` | 0.7000 | 1.0000 | 0.8154 | 0.7667 | 0.6194 | 0.7602 |
| `eligibility` | 0.6000 | 0.9000 | 0.6900 | 0.7500 | 0.5305 | 0.7301 |
| `keyword` | 0.7000 | 0.8000 | 0.4382 | 0.7143 | 0.3575 | 0.6216 |
| `lexical_diversity` | 0.7000 | 1.0000 | 0.4821 | 0.5667 | 0.4355 | 0.6300 |
| `natural_language` | 0.7000 | 0.8000 | 0.3917 | 0.6200 | 0.3931 | 0.5572 |
| `numeric` | 0.6000 | 0.8000 | 0.8118 | 0.7667 | 0.5401 | 0.5957 |
| `policy` | 0.9000 | 1.0000 | 0.6167 | 0.7067 | 0.5910 | 0.7352 |
| `procedure` | 0.9000 | 0.9000 | 0.8500 | 0.8286 | 0.6694 | 0.7808 |
| `typo` | 0.4000 | 0.7000 | 0.1456 | 0.4867 | 0.1446 | 0.4801 |

## 3. Significance: Pipeline: contextual vec + BM25, candidate_k=50 (Phase 5) vs Phase 2 pipeline: plain vec + Postgres FTS

Paired randomization test (two-sided) and 95% bootstrap CI of the per-query difference; 10,000 resamples, seed 0.

| Metric | Before | After | Δ | 95% CI | p-value | Significant (p<0.05) |
|:---|---:|---:|---:|:---:|---:|:---:|
| recall@10 | 0.6900 | 0.8700 | +0.1800 | [+0.0900, +0.2700] | 0.0006 | yes |
| mrr | 0.5615 | 0.6696 | +0.1082 | [+0.0163, +0.2013] | 0.0292 | yes |
| ndcg@10 | 0.4726 | 0.6446 | +0.1720 | [+0.1059, +0.2421] | 0.0001 | yes |
| pool_recall@50 | 0.8700 | 0.9600 | +0.0900 | [+0.0400, +0.1402] | 0.0031 | yes |

## 4. Improved queries (23)

| Query | Category | Text | MRR before | MRR after | R@10 before | R@10 after |
|:---|:---|:---|---:|---:|---:|---:|
| Q002 | `keyword` | hostel allotment | 0.500 | 1.000 | 1.00 | 1.00 |
| Q003 | `keyword` | CSE301 prerequisites | 0.500 | 1.000 | 1.00 | 1.00 |
| Q004 | `keyword` | academic transcript | 0.091 | 1.000 | 0.00 | 1.00 |
| Q006 | `natural_language` | Can I get a refund if I withdraw from the semester? | 0.333 | 1.000 | 1.00 | 1.00 |
| Q007 | `natural_language` | What are the requirements for hostel allotment? | 0.500 | 1.000 | 0.50 | 0.50 |
| Q009 | `natural_language` | How many credits can an undergraduate student register per semester? | 0.500 | 0.500 | 0.50 | 1.00 |
| Q012 | `abbreviation` | dept regulations for lab safety | 0.200 | 0.250 | 0.50 | 1.00 |
| Q013 | `abbreviation` | univ admission eligibility for international students | 0.333 | 1.000 | 1.00 | 1.00 |
| Q016 | `typo` | attendence shortage rules | 0.111 | 0.167 | 0.50 | 0.50 |
| Q017 | `typo` | hostel allotmnt process | 0.083 | 1.000 | 0.00 | 1.00 |
| Q019 | `typo` | scholarship reqmnt criteria | 0.333 | 1.000 | 1.00 | 1.00 |
| Q020 | `typo` | semester withdrawl procedure | 0.000 | 0.100 | 0.00 | 0.50 |
| Q024 | `course_code` | CSE301 practical laboratory assignments | 0.077 | 0.333 | 0.00 | 1.00 |
| Q027 | `numeric` | minimum CGPA 6.0 academic probation | 0.059 | 0.333 | 0.00 | 0.50 |
| Q028 | `numeric` | caution deposit refund within 30 days | 1.000 | 1.000 | 0.50 | 1.00 |
| Q033 | `policy` | student grievance redressal ombudsman policy | 0.333 | 0.333 | 0.50 | 1.00 |
| Q034 | `policy` | academic integrity and plagiarism prevention policy | 0.500 | 1.000 | 1.00 | 1.00 |
| Q036 | `eligibility` | who is eligible for chancellor merit scholarship | 1.000 | 1.000 | 0.50 | 1.00 |
| Q039 | `eligibility` | eligibility for student semester exchange program | 0.250 | 0.500 | 0.50 | 1.00 |
| Q040 | `eligibility` | internship NOC eligibility and credit transfer | 0.200 | 1.000 | 1.00 | 1.00 |
| Q047 | `lexical_diversity` | ExamShield portal hall ticket download and re-evaluation filing | 0.077 | 0.333 | 0.00 | 1.00 |
| Q048 | `lexical_diversity` | LibNet OPAC book reservation and account renewal guide | 0.333 | 0.500 | 1.00 | 1.00 |
| Q049 | `lexical_diversity` | campus Wi-Fi 802.1X PEAP MSCHAPv2 device authentication | 0.500 | 0.500 | 0.50 | 1.00 |

## 5. Regressed or mixed queries (10)

| Query | Category | Text | MRR before | MRR after | R@10 before | R@10 after |
|:---|:---|:---|---:|---:|---:|---:|
| Q001 | `keyword` | semester fees | 1.000 | 0.500 | 1.00 | 1.00 |
| Q005 | `keyword` | library book renewal | 0.100 | 0.071 | 0.50 | 0.00 |
| Q008 | `natural_language` | What happens if my attendance is below 75%? | 0.125 | 0.100 | 0.50 | 0.50 |
| Q015 | `abbreviation` | prof office hours and mentoring guidelines | 0.333 | 0.200 | 1.00 | 1.00 |
| Q018 | `typo` | revalution fee and deadline | 0.200 | 0.167 | 0.50 | 0.50 |
| Q021 | `course_code` | cse301 syllabus and textbooks | 1.000 | 0.500 | 0.50 | 1.00 |
| Q030 | `numeric` | late fee fine 500 rupees per week | 1.000 | 0.500 | 1.00 | 1.00 |
| Q032 | `policy` | zero tolerance anti ragging policy | 0.250 | 0.200 | 1.00 | 1.00 |
| Q038 | `eligibility` | eligibility for supplementary backlog examination | 1.000 | 0.250 | 0.50 | 1.00 |
| Q045 | `procedure` | procedure for on-duty leave application and approval | 0.250 | 0.143 | 0.50 | 0.50 |

## 6. Notes

- With 50 queries, one query moves Recall@10 by 0.01–0.02. Treat differences below ~0.03 as noise.
- BM25 indexes title, section and chunk text only; the synthetic `keywords` metadata is deliberately excluded, because it was produced by the same generator as the golden queries.
- Contextual passages change only the embedded text; the stored `chunk_text` is unchanged. Production vector search picks this up after re-running ingestion (`EMBEDDING_CONTEXTUAL=1`, the default).
