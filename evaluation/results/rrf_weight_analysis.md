# RRF Weighting Sensitivity and Attribution Analysis

**Evaluation Date:** 2026-09-25  
**Corpus Size:** 696 document chunks  
**Embedding Model:** `BAAI/bge-small-en-v1.5` (384 dimensions)  
**Total Queries:** 50 (Golden Dataset across 10 categories)  
**RRF Parameters:** $k = 60$, Vector Match Count = 20, Keyword Match Count = 20, Final $K = 20$  

---

## 1. Executive Summary & Benchmark Comparison

This analysis evaluates the sensitivity of the Full Retrieval Pipeline to the `primary_query_weight` parameter ($w \in \{1.0, 1.5, 2.0\}$) across the frozen 50-query golden benchmark.

### Aggregate Performance Table

| Configuration | Recall@5 | Recall@10 | MRR | NDCG@5 | NDCG@10 |
|:---|---:|---:|---:|---:|---:|
| **Weight 1.0 (Baseline)** | 0.5400 | 0.6900 | 0.5596 | 0.4234 | 0.4726 |
| **Weight 1.5 (Proposed)** | 0.5400 | 0.6900 | 0.5647 | 0.4256 | 0.4751 |
| **Weight 2.0 (Diagnostic)** | 0.5400 | 0.7000 | 0.5662 | 0.4301 | 0.4783 |

### Metric Deltas

| Metric Comparison | $\Delta$ Recall@5 | $\Delta$ Recall@10 | $\Delta$ MRR | $\Delta$ NDCG@5 | $\Delta$ NDCG@10 |
|:---|---:|---:|---:|---:|---:|
| **1.5 vs 1.0** | +0.0000 | +0.0000 | +0.0051 | +0.0022 | +0.0025 |
| **2.0 vs 1.0** | +0.0000 | +0.0100 | +0.0066 | +0.0067 | +0.0057 |
| **2.0 vs 1.5** | +0.0000 | +0.0100 | +0.0015 | +0.0045 | +0.0032 |

---

## 2. Query-Level Attribution Breakdown

### A. Weight 1.5 vs Weight 1.0

- **Improved Queries (4):**
  1. `Q009` (`natural_language`): MRR $0.5000 \to 1.0000$ (+0.5000), NDCG@5 $+0.1245$, NDCG@10 $+0.1245$. Target chunk `POL-COURSE-REG_CHUNK_003` promoted from rank 2 to rank 1.
  2. `Q002` (`keyword`): NDCG@10 $+0.0114$. Target chunk rank within top 10 improved.
  3. `Q019` (`typo`): NDCG@10 $+0.0089$.
  4. `Q025` (`course_code`): NDCG@5 $+0.0345$, NDCG@10 $+0.0345$.
- **Regressed Queries (4):**
  1. `Q015` (`abbreviation`): MRR $0.3333 \to 0.2000$ ($-0.1333$), NDCG@5 $-0.0382$, NDCG@10 $-0.0382$. Target chunk `ACAD-POL-006_CHUNK_003` pushed from rank 3 to rank 5 because primary query branch favored competitor chunks.
  2. `Q017` (`typo`): MRR $0.0833 \to 0.0769$ ($-0.0064$). Target chunk `HSTL-REG-001_CHUNK_001` dropped from rank 12 to rank 13.
  3. `Q027` (`numeric`): MRR $0.0588 \to 0.0000$ ($-0.0588$). Target chunk `POL-COURSE-REG_CHUNK_003` dropped from rank 17 to rank 21 (out of top 20).
  4. `Q032` (`policy`): MRR $0.2500 \to 0.2000$ ($-0.0500$), NDCG@5 $-0.0148$, NDCG@10 $-0.0148$. Target chunk `POL-ANTI-RAGGING_CHUNK_002` dropped from rank 4 to rank 5.
- **Unchanged Queries (42):**
  Remaining 42 queries show identical rankings and metrics across all evaluation criteria.

### Mathematical Reconciliation of Aggregate MRR Delta:
$$\Delta \text{MRR} = \frac{+0.5000 (\text{Q009}) - 0.1333 (\text{Q015}) - 0.0064 (\text{Q017}) - 0.0588 (\text{Q027}) - 0.0500 (\text{Q032})}{50} = \frac{+0.2515}{50} = +0.00503 \approx +0.0051$$

The reported Phase 3 aggregate gain (+0.0051 MRR) is concentrated in a **single query** (`Q009`), which masks four distinct query-level regressions.

---

### B. Weight 2.0 vs Weight 1.0

- **Improved Queries (7):**
  1. `Q003` (`keyword`): MRR $0.5000 \to 1.0000$ (+0.5000), NDCG@5 $+0.2905$, NDCG@10 $+0.2954$.
  2. `Q009` (`natural_language`): MRR $0.5000 \to 1.0000$ (+0.5000), NDCG@5 $+0.1245$, NDCG@10 $+0.1245$.
  3. `Q015` (`abbreviation`): NDCG@5 $+0.1358$ (target chunk 2 improved while target chunk 1 regressed).
  4. `Q019` (`typo`): NDCG@10 $+0.0089$.
  5. `Q025` (`course_code`): NDCG@5 $+0.0345$, NDCG@10 $+0.0345$.
  6. `Q039` (`eligibility`): MRR $0.2500 \to 0.3333$ (+0.0833), NDCG@5 $+0.0546$, NDCG@10 $+0.0546$.
  7. `Q049` (`lexical_diversity`): Recall@10 $0.5000 \to 1.0000$ (+0.5000), NDCG@10 $+0.0975$.
- **Regressed Queries (8):**
  1. `Q001` (`keyword`): MRR $1.0000 \to 0.5000$ ($-0.5000$), NDCG@5 $-0.1245$, NDCG@10 $-0.1245$.
  2. `Q017` (`typo`): MRR $0.0833 \to 0.0769$ ($-0.0064$).
  3. `Q024` (`course_code`): MRR $0.0769 \to 0.0714$ ($-0.0055$). Target chunk pushed from rank 13 to rank 14.
  4. `Q027` (`numeric`): MRR $0.0588 \to 0.0000$ ($-0.0588$). Target chunk pushed out of top 20.
  5. `Q030` (`numeric`): NDCG@10 $-0.0060$.
  6. `Q032` (`policy`): MRR $0.2500 \to 0.2000$ ($-0.0500$).
  7. `Q042` (`procedure`): NDCG@5 $-0.1660$, NDCG@10 $-0.1660$.
  8. `Q043` (`procedure`): NDCG@10 $-0.0094$.
- **Unchanged Queries (35):**
  35 queries unchanged.

---

## 3. Audit of the 10 Failure Queries

| Query ID | Category | Query Text | Weight 1.0 Target Rank | Weight 1.5 Target Rank | Weight 2.0 Target Rank | Top Competitor Chunk | Primary Retrieval Bottleneck |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Q004** | `keyword` | `academic transcript` | Not in Top 20 | Not in Top 20 | Not in Top 20 | `STU-FRM-001_CHUNK_001` | Form chunk density over statutory policy |
| **Q005** | `keyword` | `library book renewal` | Rank 10 | Rank 10 | Rank 10 | `LIB-GUI-001_CHUNK_004` | Operational guide chunks outrank policy |
| **Q008** | `natural_language` | `What happens if my attendance is below 75%?` | Rank 8 | Rank 8 | Rank 8 | `POL-ATTEND_CHUNK_005` | Within-document chunk competition |
| **Q014** | `abbreviation` | `min attendance criteria for SEE examination` | Not in Top 20 | Not in Top 20 | Not in Top 20 | `ACAD-REG-001_CHUNK_003` | Multi-query concentration on neighboring chunks |
| **Q016** | `typo` | `attendence shortage rules` | Rank 9 | Rank 9 | Rank 9 | `STU-FRM-005_CHUNK_001` | Within-document chunk competition |
| **Q017** | `typo` | `hostel allotmnt process` | Rank 12 | **Rank 13** (Regressed) | **Rank 13** (Regressed) | `HSTL-REG-001_CHUNK_003` | Typo degraded primary dense vector |
| **Q020** | `typo` | `semester withdrawl procedure` | Not in Top 20 | Not in Top 20 | Not in Top 20 | `POL-ADD-DROP_CHUNK_004` | Lexical polysemy (Add/Drop vs Tuition) |
| **Q024** | `course_code` | `CSE301 practical laboratory assignments` | Rank 13 | Rank 13 | **Rank 14** (Regressed) | `ACAD-CRS-003_CHUNK_002` | Course header chunks outrank lab chunk |
| **Q027** | `numeric` | `minimum CGPA 6.0 academic probation` | Rank 17 | **Not in Top 20** (Regressed) | **Not in Top 20** (Regressed) | `ACAD-POL-010_CHUNK_002` | Generic CGPA chunks dominate primary & variants |
| **Q047** | `lexical_diversity` | `ExamShield portal hall ticket download...` | Rank 13 | Rank 13 | Rank 13 | `EXAM-NOT-002_CHUNK_004` | Exam circulars outrank IT guide |

**Key Diagnostic Finding:** None of the 10 failure queries are resolved by increasing the primary query weight to 1.5 or 2.0. In fact, 2 queries (`Q017`, `Q027`) experienced further ranking degradation.

---

## 4. Deep-Dive Investigation of the Four Regressions

### 1. Q015 (`abbreviation`): `"prof office hours and mentoring guidelines"`
- **Ground Truth Chunks:** `ACAD-POL-006_CHUNK_002`, `ACAD-POL-006_CHUNK_003`
- **Phase 2 (w=1.0) Rank:** `CHUNK_003` at Rank 3, `CHUNK_002` at Rank 6 (MRR = 0.3333)
- **Phase 3 (w=1.5) Rank:** `CHUNK_003` at Rank 5, `CHUNK_002` at Rank 6 (MRR = 0.2000)
- **Root-Cause Attribution:**
  - Branch 0 (Primary) retrieved `ACAD-POL-006_CHUNK_004` (Rank 1) and `ACAD-POL-006_CHUNK_001` (Rank 13). Target `CHUNK_003` was **not retrieved in Branch 0 top 20** (it was retrieved only in reformulations 1, 2, 3, 4, 5).
  - Applying weight 1.5 to Branch 0 amplified competitor chunks `STU-LDR-001_CHUNK_004` and `ACAD-POL-006_CHUNK_005` (both retrieved in Branch 0), causing them to leapfrog `CHUNK_003`.
- **Verdict:** Unacceptable side effect of blind branch weighting when the primary dense retrieval misses one of the multi-aspect ground-truth chunks.

### 2. Q017 (`typo`): `"hostel allotmnt process"`
- **Ground Truth Chunks:** `HSTL-REG-001_CHUNK_001`, `HSTL-REG-001_CHUNK_002`
- **Phase 2 (w=1.0) Rank:** `CHUNK_002` at Rank 12, `CHUNK_001` at Rank 14 (MRR = 0.0833)
- **Phase 3 (w=1.5) Rank:** `CHUNK_001` at Rank 13, `CHUNK_002` at Rank 14 (MRR = 0.0769)
- **Root-Cause Attribution:**
  - The typo `"allotmnt"` caused dense vector retrieval on Branch 0 to rank generic allotment policy chunks (`HSTL-REG-001_CHUNK_003` at rank 3, `POL-HOSTEL-ALLOC_CHUNK_005` at rank 4) ahead of the specific process chunk (`CHUNK_001` at rank 14).
  - Weighting Branch 0 by 1.5 amplified the generic competitors more than the target chunk ($1.5 \times \frac{1}{63} = 0.0238$ vs $1.5 \times \frac{1}{74} = 0.0203$).
- **Verdict:** Demonstrates vulnerability of primary weighting when raw user input contains unresolved lexical errors.

### 3. Q027 (`numeric`): `"minimum CGPA 6.0 academic probation"`
- **Ground Truth Chunks:** `POL-COURSE-REG_CHUNK_003`, `POL-COURSE-REG_CHUNK_004`
- **Phase 2 (w=1.0) Rank:** `CHUNK_003` at Rank 17 (MRR = 0.0588)
- **Phase 3 (w=1.5) Rank:** `CHUNK_003` at Rank 21 / Out of Top 20 (MRR = 0.0000)
- **Root-Cause Attribution:**
  - Target `CHUNK_003` (Credit limits on academic probation) was retrieved **only by reformulations** (`Branch 1` at rank 13, `Branch 3` at rank 3) and was absent from Branch 0.
  - Generic graduation CGPA chunks (`ACAD-POL-010_CHUNK_002`, `ACAD-REG-002_CHUNK_002`) appeared in Branch 0 (ranks 1 and 3) and received the 1.5x boost, while target chunk received zero boost.
  - This pushed `CHUNK_003` completely past the top-20 boundary.
- **Verdict:** Severe negative interaction — primary weighting completely suppresses valid discoveries made exclusively by reformulation branches.

### 4. Q032 (`policy`): `"zero tolerance anti ragging policy"`
- **Ground Truth Chunks:** `POL-ANTI-RAGGING_CHUNK_001`, `POL-ANTI-RAGGING_CHUNK_002`
- **Phase 2 (w=1.0) Rank:** `CHUNK_002` at Rank 4, `CHUNK_001` at Rank 7 (MRR = 0.2500)
- **Phase 3 (w=1.5) Rank:** `CHUNK_002` at Rank 5, `CHUNK_001` at Rank 7 (MRR = 0.2000)
- **Root-Cause Attribution:**
  - In Branch 0, student discipline penalties chunk `STU-POL-004_CHUNK_001` was Rank 1 ($1.5 \times \frac{1}{61} = 0.0246$), whereas target `CHUNK_002` was Rank 9 ($1.5 \times \frac{1}{69} = 0.0217$).
  - Competitor `POL-ANTI-RAGGING_CHUNK_004` also received higher Branch 0 boost, pushing `CHUNK_002` from rank 4 to rank 5.
- **Verdict:** Marginal regression within top 10 due to intra-document competitor chunk weighting.

---

## 5. Category-Specific Benchmark Metrics

| Category | Queries | Baseline (w=1.0) MRR | Proposed (w=1.5) MRR | Diagnostic (w=2.0) MRR | Recall@10 (w=1.0) | Recall@10 (w=1.5) | Recall@10 (w=2.0) |
|:---|:---:|---:|---:|---:|---:|---:|---:|
| `abbreviation` | 5 | 0.3733 | 0.3467 | 0.3467 | 0.7000 | 0.7000 | 0.7000 |
| `course_code` | 5 | 0.8154 | 0.8154 | 0.8143 | 0.7000 | 0.7000 | 0.7000 |
| `eligibility` | 5 | 0.6900 | 0.6900 | 0.7067 | 0.6000 | 0.6000 | 0.6000 |
| `keyword` | 5 | 0.4200 | 0.4200 | 0.4200 | 0.7000 | 0.7000 | 0.7000 |
| `lexical_diversity` | 5 | 0.4821 | 0.4821 | 0.4821 | 0.7000 | 0.7000 | 0.8000 |
| `natural_language` | 5 | 0.3917 | 0.4917 | 0.4917 | 0.7000 | 0.7000 | 0.7000 |
| `numeric` | 5 | 0.8118 | 0.8000 | 0.8000 | 0.6000 | 0.6000 | 0.6000 |
| `policy` | 5 | 0.6167 | 0.6067 | 0.6067 | 0.9000 | 0.9000 | 0.9000 |
| `procedure` | 5 | 0.8500 | 0.8500 | 0.8500 | 0.9000 | 0.9000 | 0.9000 |
| `typo` | 5 | 0.1456 | 0.1443 | 0.1443 | 0.4000 | 0.4000 | 0.4000 |
| **Macro Average** | **50** | **0.5596** | **0.5647** | **0.5662** | **0.6900** | **0.6900** | **0.7000** |

All 10 categories reconcile precisely with the macro averages and per-query micro averages.
