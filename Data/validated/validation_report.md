# Enterprise Corpus Validation Report

- **Run ID:** `validation-20260920-174725-6265dc`
- **Timestamp:** 2026-09-20 17:47:26 UTC
- **Overall Status:** ❌ FAILED (Contains Errors)

## 1. Executive Summary

| Metric | Count | Description |
|---|---|---|
| **Total Documents** | 159 | Total raw institutional documents loaded |
| **Valid Documents** | 157 | Ready for downstream chunking & embedding |
| **Rejected Documents** | 2 | Excluded from validated corpus |
| **Errors** | 1 | Fatal schema, content, or ID issues |
| **Warnings** | 6 | Quality or formatting advisories |
| **Informational Notes** | 200 | Optimization and entity signals |

## 2. Duplicate Analysis

- **Exact Duplicate Groups:** 0
- **Near-Duplicate Pairs:** 0

## 3. Corpus Content & Quality Metrics

| Quality Metric | Value |
|---|---|
| Total Words | 57,725 |
| Average Word Count | 363.1 words/doc |
| Median Word Count | 347.0 words/doc |
| Min Word Count | 173 words |
| Max Word Count | 626 words |
| Average Character Count | 2610.7 chars/doc |

## 4. Corpus Distribution by Domain & Department

### By Domain

| Domain | Documents | Percentage |
|---|---|---|
| Administrative Forms & General Administration | 24 | 15.1% |
| Academic | 18 | 11.3% |
| Engineering | 16 | 10.1% |
| Examination | 15 | 9.4% |
| Student Affairs | 15 | 9.4% |
| Information Technology | 14 | 8.8% |
| Operations & Security | 12 | 7.5% |
| Hostel | 11 | 6.9% |
| Finance | 10 | 6.3% |
| Career and Placement | 9 | 5.7% |
| Research and Innovation | 8 | 5.0% |
| Library and Knowledge Resources | 7 | 4.4% |

### By Department

| Department | Documents |
|---|---|
| Academic Affairs | 21 |
| Examination Cell | 16 |
| Information Technology Services | 14 |
| Hostel Administration and Residence Life | 11 |
| Finance and Accounts Office | 10 |
| Career Development Centre / Placement Office | 9 |
| Office of Student Welfare | 9 |
| Office of Sponsored Research and Innovation | 8 |
| Department of Computer Science and Engineering | 7 |
| Central Library and Learning Resource Centre | 7 |
| Campus Safety, Vigilance and Security Department | 6 |
| Campus Operations, Estate and Facilities Management | 5 |
| Department of Student Affairs | 4 |
| Department of Electronics and Communication Engineering | 4 |
| Admissions Office / Registrar Records | 3 |
| Campus Health Centre | 3 |
| Academic Affairs & Admissions | 2 |
| Department of Mechanical Engineering | 2 |
| Department of Civil Engineering | 2 |
| Campus Mobility and Transportation Unit | 2 |
| Academic Affairs & Finance Office | 1 |
| Academic Affairs & Campus Health Centre | 1 |
| Admissions and Media Relations Office | 1 |
| Administrative Office / Registrar Secretariat | 1 |
| Admissions Office / International Wing | 1 |
| Admissions Office | 1 |
| University Governance / Registrar Secretariat | 1 |
| Academic Affairs / Registrar Secretariat | 1 |
| Academic Affairs & Career Development | 1 |
| Human Resources / Administrative Office | 1 |
| Internal Quality Assurance Cell (IQAC) | 1 |
| Office of International Relations | 1 |
| Academic Affairs & Student Affairs | 1 |
| Clubs and Student Activities Council | 1 |

## 5. Critical Errors (1)

| Document ID | Category | Field | Error Message |
|---|---|---|---|
| `POL-MEDICAL-LEAVE` | id | `document_id` | Duplicate document ID detected: 'POL-MEDICAL-LEAVE' appears 2 times across corpus |

## 6. Notable Warnings (6)

| Document ID | Category | Field | Warning Message |
|---|---|---|---|
| `SEC-SOP-001` | content | `content` | Detected placeholder or boilerplate pattern: 'XXXX' |
| `STU-GUI-001` | content | `content` | Detected placeholder or boilerplate pattern: 'XXXXX' |
| `N/A` | entity | `department` | Department ID 'DEPT-ACAD' maps inconsistently to multiple department names: ['Academic Affairs', 'Academic Affairs & Admissions', 'Academic Affairs & Campus Health Centre', 'Academic Affairs & Career Development', 'Academic Affairs & Finance Office', 'Academic Affairs & Student Affairs', 'Academic Affairs / Registrar Secretariat', 'Internal Quality Assurance Cell (IQAC)'] |
| `N/A` | entity | `department` | Department ID 'DEPT-ADM' maps inconsistently to multiple department names: ['Administrative Office / Registrar Secretariat', 'Admissions Office', 'Admissions Office / International Wing', 'Admissions Office / Registrar Records', 'Admissions and Media Relations Office', 'Human Resources / Administrative Office', 'University Governance / Registrar Secretariat'] |
| `N/A` | entity | `department` | Department ID 'DEPT-EXAM' maps inconsistently to multiple department names: ['Examination Cell', 'Information Technology Services'] |
| `IRO-POL-001` | ground_truth | `department_id` | Department ID 'DEPT-IRO' is not in authoritative world model departments |

## 7. Downstream Chunking Readiness

- **Chunking Ready:** NO - Fix Errors First
- **Target Chunker Input:** `Data/validated/validated_documents.json`
- **Document Schema Integrity:** 100% verified across required fields and structured sections.