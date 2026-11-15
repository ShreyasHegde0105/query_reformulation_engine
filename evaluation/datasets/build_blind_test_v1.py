"""Write evaluation/datasets/blind_test_v1.json (queries only; NO relevance labels).

Categories: unanswerable (from the corpus absence check), typo-heavy, multi-intent, and
precise (course codes, document IDs, named systems). Every query is marked
``label_status: "unlabelled"``. Labels must come from human annotators
(see evaluation/datasets/README.md) and are frozen with freeze_blind_test.py.

    python evaluation/datasets/check_unanswerable.py   # first
    python evaluation/datasets/build_blind_test_v1.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

AUTHORING = {
    "query_author": "AI assistant (Claude) during the 2026-11-13 audit",
    "author_saw_corpus": True,
    "note": "Not blind-authored: the author had read the corpus. Human annotators should review, "
            "replace or extend these queries before freezing.",
}

TYPO_HEAVY = [
    "hosetl gaurd pass for overnite leave", "scolorship renewel critera", "libary fien for lost buk",
    "atendance condonaton medicl certificat", "revaluaton of answr sheet fee", "plcement eligiblity cgpa cutof",
    "wifi pasword resett for hostle", "examm hall tiket downlod", "bonafied certificat for bank loan",
    "migratoin certifcate procesing time",
]
MULTI_INTENT = [
    "hostel fee refund and library fine payment", "how to apply for medical leave and who approves it",
    "bus pass renewal and route change", "exam revaluation fee and when results come out",
    "placement eligibility and internship credit rules", "ragging complaint process and punishment for ragging",
    "transfer certificate and no dues clearance steps", "wifi login for hostel and lab computer access",
    "scholarship renewal criteria and how it is credited to the fee account",
    "add drop deadline and late registration fine",
]
PRECISE = [
    ("ECE204 course outline", "course_code"), ("MEC201 lab experiments", "course_code"),
    ("CIV201 surveying practicals", "course_code"), ("CSE580 research methods", "course_code"),
    ("HSTL-REG-002", "document_id"), ("FIN-POL-003 refund rules", "document_id"),
    ("STU-FRM-006 form submission", "document_id"), ("IT-SYS-004", "document_id"),
    ("DormTrack room change request", "named_system"), ("ExamShield proctoring rules", "named_system"),
    ("LearnSphere assignment upload problem", "named_system"), ("CareerHub registration for placements", "named_system"),
]


def main() -> None:
    absence = json.load(open(os.path.join(HERE, "unanswerable_absence_check.json"), encoding="utf-8"))
    records = []

    def add(query, category, proposed, extra=None):
        records.append({
            "query_id": f"BT{len(records) + 1:03d}", "query": query, "category": category, "split": "blind_test",
            "proposed_answerability": proposed, "label_status": "unlabelled",
            "relevant_chunks": None, "graded_relevance": None, **(extra or {}),
        })

    for a in absence:
        if a["all_terms_absent"]:
            add(a["query"], "unanswerable", "proposed_unanswerable",
                {"absence_check": {t: v["chunks"] for t, v in a["term_counts"].items()}})
    for q in TYPO_HEAVY:
        add(q, "typo_heavy", "expected_answerable")
    for q in MULTI_INTENT:
        add(q, "multi_intent", "expected_answerable")
    for q, sub in PRECISE:
        add(q, f"precise_{sub}", "expected_answerable")

    out = {"version": "blind_test_v1", "authoring": AUTHORING, "labels": "NONE - awaiting human annotation",
           "queries": records}
    path = os.path.join(HERE, "blind_test_v1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    from collections import Counter
    print(path, len(records), Counter(r["category"] for r in records))


if __name__ == "__main__":
    main()
