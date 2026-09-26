# Phase 3 Regressions and Query Performance Analysis

**Date:** 2026-09-25T15:59:44.154198+00:00
**Total Queries Evaluated:** 50
**Improved Queries:** 1
**Regressed Queries:** 4
**Unchanged Queries:** 45

## 1. Regressed Queries

| Query ID | Category | Query Text | Phase 2 MRR | Phase 3 MRR | Phase 2 Rec@10 | Phase 3 Rec@10 | Probable Reason |
|:---|:---|:---|---:|---:|---:|---:|:---|
| Q015 | abbreviation | prof office hours and mentoring guidelines | 0.3333 | 0.2000 | 1.0000 | 1.0000 | Primary query weighting shift |
| Q017 | typo | hostel allotmnt process | 0.0833 | 0.0769 | 0.0000 | 0.0000 | Primary query weighting shift |
| Q027 | numeric | minimum CGPA 6.0 academic probation | 0.0588 | 0.0000 | 0.0000 | 0.0000 | Primary query weighting shift |
| Q032 | policy | zero tolerance anti ragging policy | 0.2500 | 0.2000 | 1.0000 | 1.0000 | Primary query weighting shift |

## 2. Improved Queries

| Query ID | Category | Query Text | Phase 2 MRR | Phase 3 MRR | Phase 2 Rec@10 | Phase 3 Rec@10 |
|:---|:---|:---|---:|---:|---:|---:|
| Q009 | natural_language | How many credits can an undergraduate student register per semester? | 0.5000 | 1.0000 | 0.5000 | 0.5000 |