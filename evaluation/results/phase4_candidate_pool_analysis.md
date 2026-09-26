# Phase 4 — Candidate Pool Analysis

> **Baseline**: Phase-2 Full Pipeline (weight=1.0, K=20)  

> **Question**: How many relevant chunks reach the candidate pool at K=20, K=30, K=50?


## 1. Aggregate Candidate Recall

| Pool Size | Mean Candidate Recall | ALL_PRESENT | SOME_PRESENT | NONE_PRESENT |
|-----------|----------------------|-------------|--------------|--------------|
| K=20      | 0.7800               |          31 |           16 |            3 |
| K=30      | 0.8300               |          36 |           11 |            3 |
| K=50      | 0.8600               |          38 |           10 |            2 |

## 2. Failure Query Analysis

> Phase 3 identified **10 structural failure queries**.

> Cross-encoder can address **Type B only** (relevant chunk present but ranked too low).

### Type A — Candidate Generation Failures (CE cannot help)

- **Q004**: `academic transcript`
  - K=20: recall=0.000 (NONE_PRESENT)
  - Absent chunks at K=20: `['ACAD-POL-007_CHUNK_001', 'ACAD-POL-007_CHUNK_002']`
  - K=30: recall=0.000 (NONE_PRESENT)
  - Absent chunks at K=30: `['ACAD-POL-007_CHUNK_001', 'ACAD-POL-007_CHUNK_002']`
  - K=50: recall=0.000 (NONE_PRESENT)
  - Absent chunks at K=50: `['ACAD-POL-007_CHUNK_001', 'ACAD-POL-007_CHUNK_002']`
- **Q014**: `min attendance criteria for SEE examination`
  - K=20: recall=0.000 (NONE_PRESENT)
  - Absent chunks at K=20: `['POL-ATTEND_CHUNK_002', 'POL-SEE_CHUNK_002']`
  - K=30: recall=0.000 (NONE_PRESENT)
  - Absent chunks at K=30: `['POL-ATTEND_CHUNK_002', 'POL-SEE_CHUNK_002']`
  - K=50: recall=0.000 (NONE_PRESENT)
  - Absent chunks at K=50: `['POL-ATTEND_CHUNK_002', 'POL-SEE_CHUNK_002']`

### Type B — Ranking Failures (CE may help)

- **Q005**: `library book renewal`
  - K=20: recall=0.500 (SOME_PRESENT)
  - K=30: recall=0.500 (SOME_PRESENT)
  - K=50: recall=1.000 (ALL_PRESENT)
- **Q008**: `What happens if my attendance is below 75%?`
  - K=20: recall=0.500 (SOME_PRESENT)
  - K=30: recall=0.500 (SOME_PRESENT)
  - K=50: recall=0.500 (SOME_PRESENT)
- **Q016**: `attendence shortage rules`
  - K=20: recall=0.500 (SOME_PRESENT)
  - K=30: recall=0.500 (SOME_PRESENT)
  - K=50: recall=0.500 (SOME_PRESENT)
- **Q017**: `hostel allotmnt process`
  - K=20: recall=1.000 (ALL_PRESENT)
  - K=30: recall=1.000 (ALL_PRESENT)
  - K=50: recall=1.000 (ALL_PRESENT)
- **Q020**: `semester withdrawl procedure`
  - K=20: recall=0.000 (NONE_PRESENT)
  - K=30: recall=0.000 (NONE_PRESENT)
  - K=50: recall=0.500 (SOME_PRESENT)
- **Q024**: `CSE301 practical laboratory assignments`
  - K=20: recall=0.500 (SOME_PRESENT)
  - K=30: recall=0.500 (SOME_PRESENT)
  - K=50: recall=0.500 (SOME_PRESENT)
- **Q027**: `minimum CGPA 6.0 academic probation`
  - K=20: recall=0.500 (SOME_PRESENT)
  - K=30: recall=0.500 (SOME_PRESENT)
  - K=50: recall=1.000 (ALL_PRESENT)
- **Q047**: `ExamShield portal hall ticket download and re-evaluation filing`
  - K=20: recall=0.500 (SOME_PRESENT)
  - K=30: recall=0.500 (SOME_PRESENT)
  - K=50: recall=0.500 (SOME_PRESENT)

## 3. Per-Query Detail Table

| Query | Rel | K=20 Recall | K=30 Recall | K=50 Recall | K=20 Status | K=50 Status |
|-------|-----|-------------|-------------|-------------|-------------|-------------|
| Q001 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q002 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q003 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q004 ⚠ | 2 | 0.000 | 0.000 | 0.000 | NONE_PRESENT | NONE_PRESENT |
| Q005 ⚠ | 2 | 0.500 | 0.500 | 1.000 | SOME_PRESENT | ALL_PRESENT |
| Q006 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q007 | 2 | 0.500 | 1.000 | 1.000 | SOME_PRESENT | ALL_PRESENT |
| Q008 ⚠ | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q009 | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q010 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q011 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q012 | 2 | 0.500 | 1.000 | 1.000 | SOME_PRESENT | ALL_PRESENT |
| Q013 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q014 ⚠ | 2 | 0.000 | 0.000 | 0.000 | NONE_PRESENT | NONE_PRESENT |
| Q015 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q016 ⚠ | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q017 ⚠ | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q018 | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q019 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q020 ⚠ | 2 | 0.000 | 0.000 | 0.500 | NONE_PRESENT | SOME_PRESENT |
| Q021 | 2 | 0.500 | 1.000 | 1.000 | SOME_PRESENT | ALL_PRESENT |
| Q022 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q023 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q024 ⚠ | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q025 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q026 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q027 ⚠ | 2 | 0.500 | 0.500 | 1.000 | SOME_PRESENT | ALL_PRESENT |
| Q028 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q029 | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q030 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q031 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q032 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q033 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q034 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q035 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q036 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q037 | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q038 | 2 | 0.500 | 1.000 | 1.000 | SOME_PRESENT | ALL_PRESENT |
| Q039 | 2 | 0.500 | 1.000 | 1.000 | SOME_PRESENT | ALL_PRESENT |
| Q040 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q041 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q042 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q043 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q044 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q045 | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q046 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q047 ⚠ | 2 | 0.500 | 0.500 | 0.500 | SOME_PRESENT | SOME_PRESENT |
| Q048 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q049 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |
| Q050 | 2 | 1.000 | 1.000 | 1.000 | ALL_PRESENT | ALL_PRESENT |

## 4. Key Findings

- **Mean candidate recall at K=50**: 0.8600 (vs K=20: 0.7800)

- **Type A queries** (CE powerless): 2 — ['Q004', 'Q014']

- **Type B queries** (CE may help): 8 — ['Q005', 'Q008', 'Q016', 'Q017', 'Q020', 'Q024', 'Q027', 'Q047']

- Increasing candidate pool from K=20 to K=50 may convert some Type A → Type B.

