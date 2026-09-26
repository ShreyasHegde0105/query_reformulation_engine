# Phase 3 Rigorous Audit Report — Multi-Query RRF Weighting Verification

**Audit Status:** COMPLETE  
**Audit Date:** 2026-09-25  
**Auditor:** Antigravity Advanced Agentic Engineering Team  
**Evaluation Scope:** 50-query golden dataset across 10 categories, 696 document chunks, dual-retriever hybrid backends.

---

## 1. Executive Summary & Verification of Baseline vs Reported Results

### Frozen Phase-2 Full Pipeline Baseline
| Metric | Phase 2 Baseline |
|:---|---:|
| Recall@5 | 0.5400 |
| Recall@10 | 0.6900 |
| MRR | 0.5596 |
| NDCG@5 | 0.4234 |
| NDCG@10 | 0.4726 |

### Reported Phase-3 Result
| Metric | Phase 3 Reported | Delta vs Phase 2 |
|:---|---:|:---|
| Recall@5 | 0.5400 | +0.0000 |
| Recall@10 | 0.6900 | +0.0000 |
| MRR | 0.5647 | +0.0051 |
| NDCG@5 | 0.4256 | +0.0022 |
| NDCG@10 | 0.4751 | +0.0025 |

### Audit Verification:
1. The Phase-2 baseline results are immutable and replicated exactly when `primary_query_weight = 1.0`.
2. The reported Phase-3 results are verified to be the exact mathematical result of setting `primary_query_weight = 1.5` on the current pipeline.
3. The delta of $+0.0051$ in MRR and $+0.0022$ in NDCG@5 is **100% causally driven by `primary_query_weight = 1.5`**.

---

## 2. Benchmark Configuration Audit: Execution Path & Runtime Parameter Value

A critical code audit was conducted on how `primary_query_weight` was passed during benchmark execution.

### Architectural Execution Path:
1. `SearchPipeline.search()` in `reformulation/search_pipeline.py:52` defines:
   ```python
   def search(
       self,
       query: str,
       max_variants: int = 5,
       vector_match_count: int = 20,
       keyword_match_count: int = 20,
       filter_domain: Optional[str] = None,
       k: int = 60,
       final_k: int = 20,
       primary_query_weight: float = 1.0,
   ) -> List[RetrievedChunk]:
   ```
2. In `evaluation/evaluator.py:215`, `RetrievalEvaluator.run_query_variant()` invokes `self.search_pipeline.search()`.
3. **Audit Finding:** Prior to this audit, `EvaluationConfig` in `evaluation/schema.py` did not declare `primary_query_weight`, and `evaluator.py` relied on the default parameter (`1.0`). While `run_phase3_evaluation.py` recorded `"primary_query_weight": 1.5` in its JSON metadata payload, the internal evaluator execution actually defaulted to `1.0` unless configured explicitly.
4. **Correction Applied During Audit:**
   - Updated `EvaluationConfig` to explicitly include `primary_query_weight: float = 1.0`.
   - Updated `evaluator.py:222` to pass `primary_query_weight=self.config.primary_query_weight`.
   - Updated `run_phase3_evaluation.py` to instantiate `EvaluationConfig(use_cache=True, primary_query_weight=1.5)`.
   - Preserved default API compatibility across all unit tests and external callers.

### Verified Runtime Diagnostic Configuration:
```text
Multi-query RRF configuration:
primary_query_weight = 1.5
rrf_k = 60
vector_match_count = 20
keyword_match_count = 20
final_k = 20
```

---

## 3. Controlled Three-Way Weight Experiments

The 50-query golden benchmark was executed in three controlled runs with zero other code modifications:

| Configuration | Recall@5 | Recall@10 | MRR | NDCG@5 | NDCG@10 |
|:---|---:|---:|---:|---:|---:|
| **Experiment A (Weight 1.0 - Baseline)** | 0.5400 | 0.6900 | 0.5596 | 0.4234 | 0.4726 |
| **Experiment B (Weight 1.5 - Proposed)** | 0.5400 | 0.6900 | 0.5647 | 0.4256 | 0.4751 |
| **Experiment C (Weight 2.0 - Diagnostic)** | 0.5400 | 0.7000 | 0.5662 | 0.4301 | 0.4783 |

### Metric Deltas Across Configurations:
- **1.5 vs 1.0:** $\Delta\text{Rec@5} = +0.0000$, $\Delta\text{Rec@10} = +0.0000$, $\Delta\text{MRR} = +0.0051$, $\Delta\text{NDCG@5} = +0.0022$, $\Delta\text{NDCG@10} = +0.0025$
- **2.0 vs 1.0:** $\Delta\text{Rec@5} = +0.0000$, $\Delta\text{Rec@10} = +0.0100$, $\Delta\text{MRR} = +0.0066$, $\Delta\text{NDCG@5} = +0.0067$, $\Delta\text{NDCG@10} = +0.0057$
- **2.0 vs 1.5:** $\Delta\text{Rec@5} = +0.0000$, $\Delta\text{Rec@10} = +0.0100$, $\Delta\text{MRR} = +0.0015$, $\Delta\text{NDCG@5} = +0.0045$, $\Delta\text{NDCG@3} = +0.0032$

---

## 4. Query-Level Attribution and Regression Investigation

### Query Classification (Weight 1.5 vs Weight 1.0):
- **Improved Queries (4):** `Q009` (MRR $0.5 \to 1.0$), `Q002` (NDCG@10 $+0.0114$), `Q019` (NDCG@10 $+0.0089$), `Q025` (NDCG@5/10 $+0.0345$).
- **Regressed Queries (4):** `Q015` (MRR $0.3333 \to 0.2000$), `Q017` (MRR $0.0833 \to 0.0769$), `Q027` (MRR $0.0588 \to 0.0000$), `Q032` (MRR $0.2500 \to 0.2000$).
- **Unchanged Queries (42):** Identical rankings and metrics.

### Mathematical Anatomy of the 4 Regressions:
1. **`Q015` (Mentoring guidelines & office hours):** Primary query dense retrieval retrieved competitor chunks `ACAD-POL-006_CHUNK_004` (Rank 1) and `CHUNK_001` (Rank 13). Target chunk `CHUNK_003` was only retrieved by reformulations. Boosting Branch 0 amplified competitors while giving zero boost to `CHUNK_003`, demoting it from rank 3 to rank 5.
2. **`Q017` (Hostel allotmnt process):** Typo in input query caused Branch 0 to retrieve generic hostel policy chunks ahead of process chunks. 1.5x weighting disproportionately boosted generic competitors, demoting target from rank 12 to rank 13.
3. **`Q027` (Minimum CGPA 6.0 academic probation):** Target `POL-COURSE-REG_CHUNK_003` was retrieved exclusively by reformulations (`Branch 1` and `Branch 3`) and not in `Branch 0`. Boosting `Branch 0` caused generic chunks to push `CHUNK_003` completely out of the top 20 (rank 17 to rank 21).
4. **`Q032` (Anti ragging policy):** Intra-document competitors in Branch 0 outranked target chunks, widening competitor lead when multiplied by 1.5 (demoting target from rank 4 to rank 5).

---

## 5. Audit of Previously Identified Failure Queries

| Query ID | Category | Target Rank (w=1.0) | Target Rank (w=1.5) | Target Rank (w=2.0) | Top Competitor Chunk | Failure Resolved by RRF Weighting? |
|:---|:---|:---:|:---:|:---:|:---|:---:|
| `Q004` | `keyword` | None in top 20 | None in top 20 | None in top 20 | `STU-FRM-001_CHUNK_001` | **NO** (Vector form chunk density) |
| `Q005` | `keyword` | Rank 10 | Rank 10 | Rank 10 | `LIB-GUI-001_CHUNK_004` | **NO** (Operational guide vs statutory) |
| `Q008` | `natural_language` | Rank 8 | Rank 8 | Rank 8 | `POL-ATTEND_CHUNK_005` | **NO** (Within-document detention chunk) |
| `Q014` | `abbreviation` | None in top 20 | None in top 20 | None in top 20 | `ACAD-REG-001_CHUNK_003` | **NO** (Multi-query concentration) |
| `Q016` | `typo` | Rank 9 | Rank 9 | Rank 9 | `STU-FRM-005_CHUNK_001` | **NO** (Within-document competition) |
| `Q017` | `typo` | Rank 12 | **Rank 13** | **Rank 13** | `HSTL-REG-001_CHUNK_003` | **NO** (Regressed) |
| `Q020` | `typo` | None in top 20 | None in top 20 | None in top 20 | `POL-ADD-DROP_CHUNK_004` | **NO** (Polysemy Add/Drop vs Tuition) |
| `Q024` | `course_code` | Rank 13 | Rank 13 | **Rank 14** | `ACAD-CRS-003_CHUNK_002` | **NO** (Regressed at 2.0) |
| `Q027` | `numeric` | Rank 17 | **None in top 20** | **None in top 20** | `ACAD-POL-010_CHUNK_002` | **NO** (Regressed out of top 20) |
| `Q047` | `lexical_diversity` | Rank 13 | Rank 13 | Rank 13 | `EXAM-NOT-002_CHUNK_004` | **NO** (Circular notices vs IT manual) |

---

## 6. Mathematical Verification of RRF Multi-Query Fusion

The multi-query fusion formula implemented in `SearchPipeline`:
$$\operatorname{RRF}(d) = \sum_{i=0}^{N-1} w_i \cdot \frac{1}{k + r_i(d)}, \quad \text{where } w_0 = \text{primary\_query\_weight}, \; w_{i>0} = 1.0, \; k = 60$$

### Case Verification: `Q014`
- Active branches: 6.
- Competitor `ACAD-REG-001_CHUNK_003` appeared in all 6 branches:
  $$\operatorname{Score}_{1.0} = \frac{1}{62} + \frac{1}{61} + \frac{1}{66} + \frac{1}{62} + \frac{1}{61} + \frac{1}{70} = 0.094482 \implies \text{Rank 1}$$
  $$\operatorname{Score}_{1.5} = 1.5\left(\frac{1}{62}\right) + \frac{1}{61} + \frac{1}{66} + \frac{1}{62} + \frac{1}{61} + \frac{1}{70} = 0.102547 \implies \text{Rank 1}$$
- Target `POL-SEE_CHUNK_002` appeared in 0 of 6 branches $\implies \operatorname{Score} = 0.0000 \implies \text{Unranked}$.

### Case Verification: `Q027`
- Target `POL-COURSE-REG_CHUNK_003` appeared in Branch 1 (rank 13) and Branch 3 (rank 3), but **Branch 0 rank = $\infty$**:
  $$\operatorname{Score}_{1.0} = \frac{1}{73} + \frac{1}{63} = 0.013699 + 0.015873 = 0.029572 \implies \text{Rank 17}$$
  $$\operatorname{Score}_{1.5} = 0 + \frac{1}{73} + \frac{1}{63} = 0.029572 \implies \text{Rank 21 (Overtaken by Branch 0 boosted competitors)}$$

### Case Verification: `Q047`
- Target `IT-SYS-003_CHUNK_001` appeared in Branch 0 (rank 2), Branch 1 (rank 4), Branch 3 (rank 15), Branch 4 (rank 9):
  $$\operatorname{Score}_{1.0} = \frac{1}{62} + \frac{1}{64} + \frac{1}{75} + \frac{1}{69} = 0.016129 + 0.015625 + 0.013333 + 0.014493 = 0.059580 \implies \text{Rank 13}$$
  $$\operatorname{Score}_{1.5} = 1.5\left(\frac{1}{62}\right) + \frac{1}{64} + \frac{1}{75} + \frac{1}{69} = 0.024194 + 0.015625 + 0.013333 + 0.014493 = 0.067645 \implies \text{Rank 13}$$
- Competitors `EXAM-NOT-002_CHUNK_004` and `POL-REEVAL_CHUNK_003` also appeared in Branch 0 and across all reformulations, scoring $0.081589$ and $0.081570$, maintaining their lead over the target chunk.

---

## 7. Root-Cause Hypothesis Evaluation

1. **Hypothesis 1: Multi-Query RRF Concentration Bias**
   - *Status:* **CONFIRMED**
   - *Evidence:* Chunks appearing across multiple reformulations mathematically accumulate $4\times$ to $6\times$ higher scores than specific single-branch matches, regardless of branch ranking.
2. **Hypothesis 2: Primary Query Weighting Solves Concentration Bias**
   - *Status:* **NOT SUPPORTED / REFUTED**
   - *Evidence:* When the primary query's dense retrieval is noisy or incomplete (e.g. typos, polysemy, multi-aspect queries), weighting the primary query amplifies incorrect primary competitors and suppresses valid reformulation discoveries (as demonstrated in `Q015`, `Q017`, `Q027`, `Q032`).
3. **Hypothesis 3: Reported Phase 3 Improvement is Causal from 1.5 Weight**
   - *Status:* **CONFIRMED (Isolated to Q009)**
   - *Evidence:* The aggregate $+0.0051$ MRR gain is mathematically attributable to a single query (`Q009` MRR $+0.5000 / 50 = +0.0100$) overpowering the four regressions ($-0.0049$).

---

## 8. Final Audit Recommendation

### Decision: **RETAIN 1.5 AS AN EXPERIMENTAL CONFIGURATION, BUT DO NOT RELY ON IT AS A RETRIEVAL FIX**

**Rationale:**
1. Aggregate metrics show a net neutral-to-positive score (MRR $+0.0051$, NDCG@5 $+0.0022$).
2. However, primary query weighting is a blunt global scalar:
   - It **does not solve** any of the 10 structural retrieval failure modes.
   - It introduces **4 query regressions** where reformulation branches discovered correct documents that the primary query missed.
3. Therefore, primary query weighting cannot substitute for semantic re-ranking. The path forward to solve document/chunk confusion, high-frequency form competition, and multi-query concentration bias is a **Stage-2 Cross-Encoder Re-ranker** applied over the fused candidate pool.
