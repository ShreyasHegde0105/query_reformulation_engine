# Phase 4 — Failure Query Analysis: Candidate Pool & Cross-Encoder Evaluation

> Analysis of the 10 structural failure queries identified in Phase 3.
> Evaluates candidate pool presence across K=20, K=30, K=50 and Cross-Encoder reranking capability.

## 1. Executive Summary: Type A vs Type B Classification

| Query ID | Query Text | Classification | K=20 Recall | K=50 Recall | Phase-2 MRR | CE MRR | CE Effect |
|----------|------------|----------------|-------------|-------------|-------------|--------|-----------|
| Q004 | `academic transcript` | Type B (Ranking Fail) | 0.50 | 0.50 | 0.0000 | 0.0000 | = Unchanged |
| Q005 | `library book renewal` | Type B (Ranking Fail) | 0.50 | 1.00 | 0.1000 | 0.0000 | ✗ Regressed |
| Q008 | `What happens if my attendance is below 7` | Type B (Ranking Fail) | 0.50 | 0.50 | 0.1250 | 0.5000 | ✓ Improved |
| Q014 | `min attendance criteria for SEE examinat` | Type A (Gen Fail) | 0.00 | 0.00 | 0.0000 | 0.0000 | = Unchanged |
| Q016 | `attendence shortage rules` | Type B (Ranking Fail) | 0.50 | 0.50 | 0.1111 | 0.0000 | ✗ Regressed |
| Q017 | `hostel allotmnt process` | Type B (Ranking Fail) | 1.00 | 1.00 | 0.0000 | 0.0000 | = Unchanged |
| Q020 | `semester withdrawl procedure` | Type B (Ranking Fail) | 0.00 | 0.50 | 0.0000 | 0.0000 | = Unchanged |
| Q024 | `CSE301 practical laboratory assignments` | Type B (Ranking Fail) | 0.50 | 0.50 | 0.0000 | 0.0000 | = Unchanged |
| Q027 | `minimum CGPA 6.0 academic probation` | Type B (Ranking Fail) | 0.50 | 1.00 | 0.0000 | 0.2500 | ✓ Improved |
| Q047 | `ExamShield portal hall ticket download a` | Type B (Ranking Fail) | 0.50 | 0.50 | 0.0000 | 0.0000 | = Unchanged |

---

## 2. Special Analyses for Critical Queries

### Special Analysis — Q014: `min attendance criteria for SEE examination`

- **Relevant Ground Truth Chunks**: `POL-SEE_CHUNK_002`, `POL-ATTEND_CHUNK_002`
- **Candidate Pool Presence**: K=20: 0.00, K=30: 0.00, K=50: 0.00
- **Classification**: **Type A — Candidate Generation Failure**
- **Root Cause & Branch Inspection**:
  - Neither `POL-SEE_CHUNK_002` nor `POL-ATTEND_CHUNK_002` entered the candidate pool at K=20, K=30, or K=50.
  - The multi-query branches generate attendance-related queries, but lexical mismatch with 'SEE examination' caused vector and keyword retrievers to pull general attendance forms (`STU-FRM-005`) and regulation overviews (`ACAD-REG-001`).
  - **Crucial Phase 4 Conclusion**: **STOP attributing Q014 failure to ranking.** A cross-encoder reranker can only rank candidates that exist in the candidate pool. Because candidate recall for Q014 is 0.000 across all pool sizes, second-stage reranking cannot solve Q014.

### Special Analysis — Q024: `CSE301 practical laboratory assignments`

- **Relevant Ground Truth Chunks**: `ACAD-CRS-003_CHUNK_004`, `ACAD-CRS-003_CHUNK_005`
- **Candidate Pool Presence**: Target `CHUNK_004` is present in candidate pool at RRF Rank 13; `CHUNK_005` is absent.
- **Cross-Encoder Ranking Behavior**:
  - `ACAD-CRS-003_CHUNK_001` (Course Syllabus & Overview) scored CE: **+5.0922** (Rank 1).
  - `ACAD-CRS-003_CHUNK_004` (Lab Experiments & Assignments) scored CE: **-2.4721** (Rank 17).
- **Root Cause**:
  - While `CHUNK_004` is the actual lab assignment text, `CHUNK_001` prominently features the course title `CSE301: Operating Systems (OS) Syllabus and System Programming Lab` in its header.
  - The cross-encoder strongly correlates title-level course code + keyword overlap, ranking the syllabus header chunk higher than the specific assignment chunk.

### Special Analysis — Q004: `academic transcript`

- **Relevant Ground Truth Chunks**: `EXAM-GUI-005_CHUNK_002`, `EXAM-GUI-005_CHUNK_001`
- **Label Correction**: Earlier runs labelled this query against `ACAD-POL-007` (Graduate Teaching Assistantship Norms), which never mentions transcripts. Relabelled to `EXAM-GUI-005` (Examination Cell counters for grade reports, marks cards and transcript attestation).
- **Candidate Pool Presence**: `EXAM-GUI-005_CHUNK_002` at RRF rank 11 → CE rank 16 (CE score -6.9287).
- **Classification**: **Type B (Ranking Failure - Partial Present in pool)**
- **Top RRF Competitors**: `STU-FRM-001_CHUNK_001`, `STU-FRM-002_CHUNK_001`, `STU-FRM-003_CHUNK_001` — administrative forms and graduation policies that list transcripts as a prerequisite outrank the Examination Cell directory.

### Special Analysis — Q027: `minimum CGPA 6.0 academic probation`

- **Relevant Ground Truth Chunks**: `POL-COURSE-REG_CHUNK_003`, `POL-COURSE-REG_CHUNK_004`
- **Candidate Pool Presence**: Both targets present in K=50 pool (`CHUNK_003` RRF rank 17, `CHUNK_004` RRF rank 45).
- **Cross-Encoder Ranking Correction**:
  - In RRF, `POL-COURSE-REG_CHUNK_003` was suppressed to **rank 17** by general graduation policies.
  - The Cross-Encoder scored `CHUNK_003` at **+2.4372**, promoting it from **rank 17 to rank 4**!
  - **Outcome**: **Successful ranking correction (Type B fix)**. Demonstrates clear evidence that cross-encoder correctly identifies query-relevant policy details when present in candidate pool.

### Special Analysis — Q047: `ExamShield portal hall ticket download and re-evaluation filing`

- **Relevant Ground Truth Chunks**: `IT-SYS-003_CHUNK_001`, `IT-SYS-003_CHUNK_002`
- **Candidate Pool Presence**: `IT-SYS-003_CHUNK_001` present at RRF rank 13; `CHUNK_002` absent.
- **Cross-Encoder Discrimination Analysis**:
  - Competing chunks: Generic examination notices (`EXAM-NOT-002`, `EXAM-NOT-003`) vs IT system guide (`IT-SYS-003`).
  - `IT-SYS-003_CHUNK_003` (ExamShield guide) ranked high in CE (+3.8125, Rank 3).
  - `POL-SEE_CHUNK_004` ranked CE 1 (+4.9897) due to 'evaluation filing' and 'hall ticket' lexical overlap.
  - Target `IT-SYS-003_CHUNK_001` received CE score -0.8361 (Rank 18).

---

## 3. Complete Per-Query Candidate Pool & Ranking Tables (The 10 Failure Queries)

### Q004: `academic transcript`

**Classification**: Type B (Ranking Failure - Partial Present in pool)

**Relevant Ground Truth**: `['EXAM-GUI-005_CHUNK_002', 'EXAM-GUI-005_CHUNK_001']`

**Pool Recall**: K=20: 0.50 | K=30: 0.50 | K=50: 0.50

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `STU-FRM-001_CHUNK_001` | Application Form: General Bonafide Stude | 0.05710 | -5.9972 | 8 | no |
| 2 | `STU-FRM-002_CHUNK_001` | Application Form: Education Loan Fee Est | 0.04563 | -7.8955 | 23 | no |
| 3 | `STU-FRM-003_CHUNK_001` | Application Form: Transfer Certificate ( | 0.04153 | -7.0968 | 17 | no |
| 4 | `STU-FRM-003_CHUNK_004` | Application Form: Transfer Certificate ( | 0.04059 | -9.3461 | 33 | no |
| 5 | `POL-GRADUATION_CHUNK_005` | Degree Award and Graduation Requirements | 0.03226 | -1.9293 | 3 | no |
| 6 | `ACAD-POL-010_CHUNK_005` | Graduation Requirements and Minimum CGPA | 0.03132 | 1.1283 | 1 | no |
| 7 | `ADM-POL-004_CHUNK_004` | Procedure for Withdrawal, Transfer Certi | 0.03068 | -9.3739 | 34 | no |
| 8 | `STU-FRM-001_CHUNK_003` | Application Form: General Bonafide Stude | 0.03009 | -10.3501 | 42 | no |
| 9 | `FIN-POL-005_CHUNK_004` | Installment Payment Scheme Guidelines fo | 0.02924 | -10.5743 | 44 | no |
| 10 | `EXAM-NOT-002_CHUNK_001` | Circular: Supplementary Backlog Examinat | 0.02896 | -8.9971 | 28 | no |
| 11 | `EXAM-GUI-005_CHUNK_002` | Examination Cell Counter Directory and O | 0.02889 | -6.9287 | 16 | **✓ YES** |
| 12 | `STU-FRM-002_CHUNK_003` | Application Form: Education Loan Fee Est | 0.02848 | -10.6038 | 46 | no |
| 13 | `POL-TRANSFER-CERT_CHUNK_002` | Transfer Certificate (TC), Migration, an | 0.02775 | -8.3593 | 27 | no |
| 14 | `PLC-GUI-001_CHUNK_002` | CareerHub Portal Student Guide: Drive Re | 0.02760 | -5.7005 | 5 | no |
| 15 | `STU-FRM-005_CHUNK_003` | Application Form: Attendance Shortage Me | 0.02721 | -9.5759 | 37 | no |


### Q005: `library book renewal`

**Classification**: Type B (Ranking Failure - All Present in pool)

**Relevant Ground Truth**: `['LIB-POL-001_CHUNK_002', 'LIB-POL-001_CHUNK_003']`

**Pool Recall**: K=20: 0.50 | K=30: 0.50 | K=50: 1.00

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `LIB-GUI-001_CHUNK_004` | LibNet OPAC Search, Book Reservation, an | 0.08197 | 6.2201 | 1 | no |
| 2 | `LIB-POL-001_CHUNK_004` | Central Library Borrowing Rules, Circula | 0.08039 | -2.2961 | 6 | no |
| 3 | `LIB-FAQ-001_CHUNK_001` | Central Library Services, Off-Campus E-A | 0.07962 | -3.7494 | 7 | no |
| 4 | `LIB-FAQ-001_CHUNK_002` | Central Library Services, Off-Campus E-A | 0.07581 | -0.8453 | 4 | no |
| 5 | `LIB-POL-001_CHUNK_005` | Central Library Borrowing Rules, Circula | 0.07429 | -0.5519 | 3 | no |
| 6 | `LIB-GUI-001_CHUNK_003` | LibNet OPAC Search, Book Reservation, an | 0.07421 | -5.3004 | 9 | no |
| 7 | `LIB-GOV-001_CHUNK_004` | Central Library Advisory Committee Const | 0.07376 | 1.7407 | 2 | no |
| 8 | `FIN-POL-002_CHUNK_003` | Incidental Administrative Fees, Challan  | 0.07126 | -9.0642 | 18 | no |
| 9 | `LIB-POL-001_CHUNK_006` | Central Library Borrowing Rules, Circula | 0.06903 | -9.3008 | 19 | no |
| 10 | `LIB-POL-001_CHUNK_003` | Central Library Borrowing Rules, Circula | 0.06040 | -6.5424 | 12 | **✓ YES** |
| 11 | `LIB-POL-001_CHUNK_001` | Central Library Borrowing Rules, Circula | 0.05786 | -1.3435 | 5 | no |
| 12 | `LIB-GUI-001_CHUNK_002` | LibNet OPAC Search, Book Reservation, an | 0.05591 | -9.9260 | 23 | no |
| 13 | `FIN-POL-003_CHUNK_003` | Caution Deposit Refund and Account Closu | 0.05281 | -6.8118 | 14 | no |
| 14 | `FIN-POL-003_CHUNK_004` | Caution Deposit Refund and Account Closu | 0.05269 | -9.5287 | 20 | no |
| 15 | `LIB-GUI-001_CHUNK_001` | LibNet OPAC Search, Book Reservation, an | 0.04266 | -8.0746 | 16 | no |


### Q008: `What happens if my attendance is below 75%?`

**Classification**: Type B (Ranking Failure - Partial Present in pool)

**Relevant Ground Truth**: `['POL-ATTEND_CHUNK_002', 'POL-ATTEND_CHUNK_003']`

**Pool Recall**: K=20: 0.50 | K=30: 0.50 | K=50: 0.50

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `POL-ATTEND_CHUNK_005` | Mandatory Attendance and Condonation Pol | 0.09606 | 0.0449 | 3 | no |
| 2 | `STU-FRM-005_CHUNK_001` | Application Form: Attendance Shortage Me | 0.09558 | -2.9261 | 8 | no |
| 3 | `ACAD-REG-001_CHUNK_003` | General Academic Regulations for Autonom | 0.09064 | 1.9657 | 1 | no |
| 4 | `POL-ATTEND_CHUNK_004` | Mandatory Attendance and Condonation Pol | 0.08911 | -4.4205 | 11 | no |
| 5 | `PLC-TRN-001_CHUNK_004` | CPTD Pre-Placement Training Bootcamps an | 0.08740 | -2.8563 | 7 | no |
| 6 | `ACAD-POL-004_CHUNK_003` | On-Duty (OD) Sanction and Attendance Adj | 0.08555 | -5.3862 | 14 | no |
| 7 | `ACAD-POL-003_CHUNK_001` | Policy on Attendance Condonation on Medi | 0.07528 | -0.3833 | 5 | no |
| 8 | `POL-ATTEND_CHUNK_003` | Mandatory Attendance and Condonation Pol | 0.07462 | 1.4274 | 2 | **✓ YES** |
| 9 | `ACAD-POL-003_CHUNK_002` | Policy on Attendance Condonation on Medi | 0.07358 | -1.3898 | 6 | no |
| 10 | `POL-SEE_CHUNK_004` | Semester End Examination (SEE) and Evalu | 0.07141 | -5.1306 | 13 | no |
| 11 | `STU-FRM-004_CHUNK_003` | Application Form: On-Duty (OD) Leave San | 0.06806 | -11.0334 | 28 | no |
| 12 | `STU-FRM-005_CHUNK_002` | Application Form: Attendance Shortage Me | 0.05870 | -0.1350 | 4 | no |
| 13 | `FIN-SCH-001_CHUNK_003` | University Merit and Need-Based Financia | 0.05462 | -11.1635 | 33 | no |
| 14 | `PLC-POL-002_CHUNK_004` | Student Internship Guidelines and Corpor | 0.05310 | -10.6321 | 22 | no |
| 15 | `STU-LDR-001_CHUNK_005` | Class Representative (CR) Election, Role | 0.04247 | -11.1737 | 35 | no |


### Q014: `min attendance criteria for SEE examination`

**Classification**: Type A (Candidate Generation Failure - Absent from pool)

**Relevant Ground Truth**: `['POL-SEE_CHUNK_002', 'POL-ATTEND_CHUNK_002']`

**Pool Recall**: K=20: 0.00 | K=30: 0.00 | K=50: 0.00

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `ACAD-REG-001_CHUNK_003` | General Academic Regulations for Autonom | 0.09448 | -1.2189 | 10 | no |
| 2 | `ACAD-POL-004_CHUNK_003` | On-Duty (OD) Sanction and Attendance Adj | 0.09116 | -0.9796 | 8 | no |
| 3 | `POL-SEE_CHUNK_003` | Semester End Examination (SEE) and Evalu | 0.08973 | 1.3146 | 4 | no |
| 4 | `POL-SEE_CHUNK_004` | Semester End Examination (SEE) and Evalu | 0.08918 | 1.8709 | 2 | no |
| 5 | `POL-ATTEND_CHUNK_005` | Mandatory Attendance and Condonation Pol | 0.08882 | -2.7148 | 13 | no |
| 6 | `FIN-POL-005_CHUNK_002` | Installment Payment Scheme Guidelines fo | 0.08457 | -5.9159 | 24 | no |
| 7 | `STU-FRM-005_CHUNK_001` | Application Form: Attendance Shortage Me | 0.08306 | -4.6168 | 22 | no |
| 8 | `POL-ATTEND_CHUNK_003` | Mandatory Attendance and Condonation Pol | 0.07585 | -4.2344 | 20 | no |
| 9 | `ACAD-REG-001_CHUNK_004` | General Academic Regulations for Autonom | 0.07354 | 1.7044 | 3 | no |
| 10 | `POL-OD_CHUNK_002` | On-Duty (OD) Leave and Co-Curricular Abs | 0.07273 | -2.8696 | 14 | no |
| 11 | `EXAM-NOT-003_CHUNK_002` | Notice: Hall Ticket Release and Biometri | 0.07187 | -3.1631 | 15 | no |
| 12 | `EXAM-NOT-005_CHUNK_004` | Notification: Fast-Track Summer Term Exa | 0.07151 | 2.7230 | 1 | no |
| 13 | `POL-CIE_CHUNK_004` | Continuous Internal Evaluation (CIE) Pol | 0.06964 | 1.2889 | 5 | no |
| 14 | `FIN-SCH-001_CHUNK_003` | University Merit and Need-Based Financia | 0.06709 | -9.3220 | 34 | no |
| 15 | `POL-ATTEND_CHUNK_001` | Mandatory Attendance and Condonation Pol | 0.05389 | -6.8896 | 28 | no |


### Q016: `attendence shortage rules`

**Classification**: Type B (Ranking Failure - Partial Present in pool)

**Relevant Ground Truth**: `['POL-ATTEND_CHUNK_002', 'POL-ATTEND_CHUNK_003']`

**Pool Recall**: K=20: 0.50 | K=30: 0.50 | K=50: 0.50

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `STU-FRM-005_CHUNK_001` | Application Form: Attendance Shortage Me | 0.07868 | -5.3073 | 5 | no |
| 2 | `ACAD-POL-004_CHUNK_003` | On-Duty (OD) Sanction and Attendance Adj | 0.07632 | -4.9491 | 4 | no |
| 3 | `POL-ATTEND_CHUNK_005` | Mandatory Attendance and Condonation Pol | 0.07515 | -3.0942 | 2 | no |
| 4 | `ACAD-POL-003_CHUNK_001` | Policy on Attendance Condonation on Medi | 0.07484 | -7.9890 | 15 | no |
| 5 | `ACAD-REG-001_CHUNK_003` | General Academic Regulations for Autonom | 0.07326 | -3.8749 | 3 | no |
| 6 | `POL-ATTEND_CHUNK_004` | Mandatory Attendance and Condonation Pol | 0.07314 | 0.6022 | 1 | no |
| 7 | `POL-OD_CHUNK_002` | On-Duty (OD) Leave and Co-Curricular Abs | 0.07008 | -7.4693 | 12 | no |
| 8 | `POL-ATTEND_CHUNK_001` | Mandatory Attendance and Condonation Pol | 0.05840 | -5.6164 | 6 | no |
| 9 | `POL-ATTEND_CHUNK_003` | Mandatory Attendance and Condonation Pol | 0.05732 | -7.3584 | 11 | **✓ YES** |
| 10 | `STU-FRM-004_CHUNK_003` | Application Form: On-Duty (OD) Leave San | 0.05590 | -5.6328 | 7 | no |
| 11 | `POL-SEE_CHUNK_004` | Semester End Examination (SEE) and Evalu | 0.04387 | -8.8354 | 21 | no |
| 12 | `ACAD-POL-003_CHUNK_002` | Policy on Attendance Condonation on Medi | 0.04352 | -6.1904 | 8 | no |
| 13 | `STU-FRM-005_CHUNK_002` | Application Form: Attendance Shortage Me | 0.04299 | -6.8629 | 9 | no |
| 14 | `FIN-SCH-001_CHUNK_003` | University Merit and Need-Based Financia | 0.04215 | -10.5375 | 33 | no |
| 15 | `STU-LDR-001_CHUNK_005` | Class Representative (CR) Election, Role | 0.04186 | -10.3944 | 29 | no |


### Q017: `hostel allotmnt process`

**Classification**: Type B (Ranking Failure - All Present in pool)

**Relevant Ground Truth**: `['HSTL-REG-001_CHUNK_001', 'HSTL-REG-001_CHUNK_002']`

**Pool Recall**: K=20: 1.00 | K=30: 1.00 | K=50: 1.00

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `HSTL-REG-001_CHUNK_003` | Hostel Admission, Room Allotment Matrix, | 0.09526 | -4.8615 | 9 | no |
| 2 | `POL-HOSTEL-ALLOC_CHUNK_005` | Student Residential Housing and Hostel A | 0.09314 | -4.6901 | 8 | no |
| 3 | `POL-HOSTEL-ALLOC_CHUNK_001` | Student Residential Housing and Hostel A | 0.08620 | -3.6223 | 7 | no |
| 4 | `HSTL-POL-005_CHUNK_002` | Off-Campus Student Housing & Verified PG | 0.08545 | -7.8483 | 26 | no |
| 5 | `LIB-POL-003_CHUNK_003` | Reference Hall, 24/7 Reading Room, and D | 0.08495 | -5.6159 | 15 | no |
| 6 | `POL-HOSTEL-ALLOC_CHUNK_003` | Student Residential Housing and Hostel A | 0.08197 | 3.1619 | 1 | no |
| 7 | `HSTL-POL-004_CHUNK_002` | Hostel Room Vacation, Inventory Clearanc | 0.08060 | -7.7103 | 25 | no |
| 8 | `HSTL-REG-002_CHUNK_001` | Hostel Operations Division Maintenance S | 0.07515 | -2.9210 | 5 | no |
| 9 | `HSTL-REG-002_CHUNK_004` | Hostel Operations Division Maintenance S | 0.07474 | -3.2791 | 6 | no |
| 10 | `HSTL-REG-004_CHUNK_003` | Residence Hall Leadership and Resident A | 0.06980 | -2.7140 | 4 | no |
| 11 | `HSTL-REG-002_CHUNK_002` | Hostel Operations Division Maintenance S | 0.06032 | -5.1479 | 12 | no |
| 12 | `HSTL-REG-001_CHUNK_002` | Hostel Admission, Room Allotment Matrix, | 0.05928 | -7.0979 | 22 | **✓ YES** |
| 13 | `HSTL-REG-001_CHUNK_005` | Hostel Admission, Room Allotment Matrix, | 0.05876 | -7.0955 | 21 | no |
| 14 | `HSTL-REG-001_CHUNK_001` | Hostel Admission, Room Allotment Matrix, | 0.05614 | -5.0225 | 11 | **✓ YES** |
| 15 | `POL-ANTI-RAGGING_CHUNK_005` | Zero-Tolerance Anti-Ragging and Harassme | 0.04668 | -5.8744 | 17 | no |


### Q020: `semester withdrawl procedure`

**Classification**: Type B (Ranking Failure - Partial Present in pool)

**Relevant Ground Truth**: `['FIN-POL-004_CHUNK_002', 'FIN-POL-004_CHUNK_003']`

**Pool Recall**: K=20: 0.00 | K=30: 0.00 | K=50: 0.50

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `POL-ADD-DROP_CHUNK_004` | Course Add/Drop and Course Withdrawal Po | 0.09757 | 3.1831 | 1 | no |
| 2 | `FIN-POL-004_CHUNK_004` | Program Withdrawal and Tuition Fee Refun | 0.09528 | -3.9414 | 8 | no |
| 3 | `FIN-POL-003_CHUNK_003` | Caution Deposit Refund and Account Closu | 0.08644 | -5.7843 | 16 | no |
| 4 | `HSTL-POL-004_CHUNK_001` | Hostel Room Vacation, Inventory Clearanc | 0.07755 | 0.2953 | 2 | no |
| 5 | `POL-ADD-DROP_CHUNK_001` | Course Add/Drop and Course Withdrawal Po | 0.07486 | -0.2023 | 4 | no |
| 6 | `POL-ADD-DROP_CHUNK_003` | Course Add/Drop and Course Withdrawal Po | 0.07158 | -6.5014 | 21 | no |
| 7 | `ADM-SOP-001_CHUNK_001` | Comprehensive University Institutional N | 0.07123 | -0.3867 | 5 | no |
| 8 | `FIN-POL-004_CHUNK_001` | Program Withdrawal and Tuition Fee Refun | 0.06953 | -6.1698 | 20 | no |
| 9 | `FIN-SCH-001_CHUNK_003` | University Merit and Need-Based Financia | 0.06891 | -5.6428 | 14 | no |
| 10 | `ADM-POL-004_CHUNK_001` | Procedure for Withdrawal, Transfer Certi | 0.06838 | -3.7767 | 7 | no |
| 11 | `POL-SEE_CHUNK_001` | Semester End Examination (SEE) and Evalu | 0.06070 | -4.2030 | 9 | no |
| 12 | `ADM-SOP-001_CHUNK_003` | Comprehensive University Institutional N | 0.05930 | -10.1283 | 34 | no |
| 13 | `POL-ADD-DROP_CHUNK_005` | Course Add/Drop and Course Withdrawal Po | 0.05879 | -0.0625 | 3 | no |
| 14 | `STU-FRM-003_CHUNK_003` | Application Form: Transfer Certificate ( | 0.05414 | -10.7985 | 41 | no |
| 15 | `LIB-POL-001_CHUNK_006` | Central Library Borrowing Rules, Circula | 0.05179 | -5.1601 | 11 | no |


### Q024: `CSE301 practical laboratory assignments`

**Classification**: Type B (Ranking Failure - Partial Present in pool)

**Relevant Ground Truth**: `['ACAD-CRS-003_CHUNK_004', 'ACAD-CRS-003_CHUNK_005']`

**Pool Recall**: K=20: 0.50 | K=30: 0.50 | K=50: 0.50

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `ACAD-CRS-003_CHUNK_002` | CSE301: Operating Systems (OS) Syllabus  | 0.09836 | 1.2241 | 8 | no |
| 2 | `ACAD-CRS-003_CHUNK_001` | CSE301: Operating Systems (OS) Syllabus  | 0.09550 | 5.0922 | 1 | no |
| 3 | `ACAD-CRS-005_CHUNK_002` | CSE305: Software Engineering (SE) Agile  | 0.09481 | 1.0114 | 9 | no |
| 4 | `ACAD-CRS-005_CHUNK_001` | CSE305: Software Engineering (SE) Agile  | 0.09051 | 0.3625 | 11 | no |
| 5 | `ACAD-CRS-004_CHUNK_002` | CSE303: Database Management Systems (DBM | 0.08997 | 3.9075 | 3 | no |
| 6 | `ACAD-CRS-004_CHUNK_001` | CSE303: Database Management Systems (DBM | 0.08755 | 4.6299 | 2 | no |
| 7 | `ACAD-CRS-006_CHUNK_001` | CSE401: Artificial Intelligence and Mach | 0.08739 | 3.5114 | 4 | no |
| 8 | `ACAD-CRS-002_CHUNK_002` | CSE202: Object-Oriented Programming with | 0.08646 | -4.5348 | 23 | no |
| 9 | `ACAD-CRS-002_CHUNK_001` | CSE202: Object-Oriented Programming with | 0.08638 | -0.5343 | 12 | no |
| 10 | `CSE-CRS-304_CHUNK_001` | Computer Networks and IP Routing Protoco | 0.08475 | 2.2942 | 6 | no |
| 11 | `ACAD-CRS-006_CHUNK_002` | CSE401: Artificial Intelligence and Mach | 0.07195 | 0.9091 | 10 | no |
| 12 | `ACAD-CRS-008_CHUNK_002` | ECE302: VLSI Design and Microelectronics | 0.06629 | -1.2509 | 14 | no |
| 13 | `ACAD-CRS-003_CHUNK_004` | CSE301: Operating Systems (OS) Syllabus  | 0.05836 | -2.4721 | 17 | **✓ YES** |
| 14 | `ACAD-CRS-001_CHUNK_002` | CSE201: Data Structures and Algorithms ( | 0.05465 | -11.2590 | 39 | no |
| 15 | `CSE-CRS-304_CHUNK_002` | Computer Networks and IP Routing Protoco | 0.05147 | -5.1766 | 25 | no |


### Q027: `minimum CGPA 6.0 academic probation`

**Classification**: Type B (Ranking Failure - All Present in pool)

**Relevant Ground Truth**: `['POL-COURSE-REG_CHUNK_003', 'POL-COURSE-REG_CHUNK_004']`

**Pool Recall**: K=20: 0.50 | K=30: 0.50 | K=50: 1.00

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `ACAD-POL-010_CHUNK_002` | Graduation Requirements and Minimum CGPA | 0.07989 | 1.6455 | 8 | no |
| 2 | `ACAD-REG-002_CHUNK_002` | Academic Regulations for Postgraduate Pr | 0.07951 | 3.8136 | 1 | no |
| 3 | `POL-ATTEND_CHUNK_005` | Mandatory Attendance and Condonation Pol | 0.07472 | -3.4830 | 16 | no |
| 4 | `ACAD-POL-001_CHUNK_002` | Regulations on Course Registration, Add/ | 0.07403 | 0.6352 | 10 | no |
| 5 | `HSTL-REG-004_CHUNK_002` | Residence Hall Leadership and Resident A | 0.06202 | 3.5719 | 2 | no |
| 6 | `IRO-POL-001_CHUNK_003` | Semester Exchange and International Dual | 0.06140 | -1.7992 | 13 | no |
| 7 | `FIN-SCH-001_CHUNK_003` | University Merit and Need-Based Financia | 0.06064 | 2.5528 | 3 | no |
| 8 | `ACAD-POL-008_CHUNK_002` | Policy on Industry Internship-Based 8th  | 0.06064 | -0.1772 | 11 | no |
| 9 | `PLC-GUI-001_CHUNK_004` | CareerHub Portal Student Guide: Drive Re | 0.05633 | -7.2768 | 28 | no |
| 10 | `PLC-POL-001_CHUNK_002` | Campus Recruitment Policy, CDC Registrat | 0.04437 | 1.9862 | 7 | no |
| 11 | `STU-LDR-001_CHUNK_002` | Class Representative (CR) Election, Role | 0.04394 | -1.8423 | 14 | no |
| 12 | `POL-SEE_CHUNK_003` | Semester End Examination (SEE) and Evalu | 0.04148 | -5.5571 | 21 | no |
| 13 | `EXAM-GUI-001_CHUNK_004` | Regulations Governing Continuous Interna | 0.04130 | -9.1825 | 37 | no |
| 14 | `PLC-FAQ-001_CHUNK_002` | Frequently Asked Questions Regarding Cam | 0.04055 | 2.0371 | 5 | no |
| 15 | `PLC-GUI-001_CHUNK_003` | CareerHub Portal Student Guide: Drive Re | 0.04040 | -7.1818 | 27 | no |


### Q047: `ExamShield portal hall ticket download and re-evaluation filing`

**Classification**: Type B (Ranking Failure - Partial Present in pool)

**Relevant Ground Truth**: `['IT-SYS-003_CHUNK_001', 'IT-SYS-003_CHUNK_002']`

**Pool Recall**: K=20: 0.50 | K=30: 0.50 | K=50: 0.50

| Pool Rank | Chunk ID | Title | RRF Score | CE Score | CE Rank | Ground Truth |
|-----------|----------|-------|-----------|----------|---------|--------------|
| 1 | `EXAM-NOT-002_CHUNK_004` | Circular: Supplementary Backlog Examinat | 0.07464 | 0.6095 | 12 | no |
| 2 | `EXAM-GUI-004_CHUNK_002` | Procedure for Answer Script Viewing, Re- | 0.07391 | 0.2654 | 15 | no |
| 3 | `POL-REEVAL_CHUNK_003` | Answer Script Viewing, Re-Totalling, and | 0.07376 | 1.2647 | 10 | no |
| 4 | `POL-BACKLOG_CHUNK_003` | Backlog Arrears, Supplementary Exams, an | 0.07187 | 3.1954 | 7 | no |
| 5 | `FIN-POL-002_CHUNK_002` | Incidental Administrative Fees, Challan  | 0.07180 | -0.8896 | 20 | no |
| 6 | `EXAM-GUI-002_CHUNK_002` | Semester End Examination (SEE) Conduct,  | 0.06976 | -2.1129 | 22 | no |
| 7 | `STU-FRM-006_CHUNK_002` | Application Form: SEE Answer Script View | 0.06568 | -0.3158 | 16 | no |
| 8 | `IT-SYS-003_CHUNK_003` | ExamShield Portal Guide: Hall Ticket Dow | 0.06538 | 3.8125 | 3 | no |
| 9 | `EXAM-NOT-002_CHUNK_005` | Circular: Supplementary Backlog Examinat | 0.06531 | 3.2830 | 6 | no |
| 10 | `EXAM-GUI-005_CHUNK_002` | Examination Cell Counter Directory and O | 0.06520 | 3.6366 | 4 | no |
| 11 | `EXAM-NOT-003_CHUNK_001` | Notice: Hall Ticket Release and Biometri | 0.06212 | 4.5160 | 2 | no |
| 12 | `POL-SEE_CHUNK_004` | Semester End Examination (SEE) and Evalu | 0.06182 | 4.9897 | 1 | no |
| 13 | `IT-SYS-003_CHUNK_001` | ExamShield Portal Guide: Hall Ticket Dow | 0.05958 | -0.8361 | 18 | **✓ YES** |
| 14 | `EXAM-NOT-001_CHUNK_004` | Official Notification: Fall Semester End | 0.05922 | 0.9656 | 11 | no |
| 15 | `POL-SEE_CHUNK_005` | Semester End Examination (SEE) and Evalu | 0.05603 | 2.5908 | 8 | no |


