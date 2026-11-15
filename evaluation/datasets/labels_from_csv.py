"""Convert completed human annotation CSVs into blind_test_v1_labels.json.

Refuses to write a label file unless every pooled row and every query row has been filled in
by a named annotator (grade 0-3 / answerability, annotator_id, annotated_at). It never fills
in missing grades. Annotator metadata (role, organisation) comes from annotators.json:

    [{"annotator_id": "A1", "role": "examination office", "organisation": "Demo University"}]

    python evaluation/datasets/labels_from_csv.py --annotators evaluation/datasets/annotators.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANSWERABILITY = {"fully_answerable", "partially_answerable", "unanswerable"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", required=True)
    ap.add_argument("--pool", default=os.path.join(HERE, "blind_test_v1_annotation_pool.csv"))
    ap.add_argument("--answerability", default=os.path.join(HERE, "blind_test_v1_answerability.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "blind_test_v1_labels.json"))
    args = ap.parse_args()

    annotators = json.load(open(args.annotators, encoding="utf-8"))
    known = {a["annotator_id"] for a in annotators}
    errors = []
    queries = {}
    for row in csv.DictReader(open(args.answerability, encoding="utf-8")):
        if row["answerability"] not in ANSWERABILITY or row["annotator_id"] not in known or not row["annotated_at"]:
            errors.append(f"answerability row {row['query_id']} incomplete")
            continue
        queries[row["query_id"]] = {"answerability": row["answerability"], "answerability_annotator": row["annotator_id"],
                                    "judgments": []}
    for row in csv.DictReader(open(args.pool, encoding="utf-8")):
        if row["grade"] not in {"0", "1", "2", "3"} or row["annotator_id"] not in known or not row["annotated_at"]:
            errors.append(f"pool row {row['query_id']}/{row['chunk_id']} incomplete")
            continue
        if row["query_id"] in queries:
            queries[row["query_id"]]["judgments"].append({
                "chunk_id": row["chunk_id"], "grade": int(row["grade"]), "annotator_id": row["annotator_id"],
                "annotated_at": row["annotated_at"], "notes": row.get("notes", "")})
    if errors:
        print(f"Not written: {len(errors)} incomplete rows, e.g. {errors[:5]}")
        return 1
    out = {"label_source": "human", "annotators": annotators,
           "guidelines": "evaluation/judge/rubric.py relevance-rubric-v1 (grades 0-3)", "queries": queries}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {args.out}: {len(queries)} queries. Next: python evaluation/datasets/freeze_blind_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
