"""Freeze human blind-test labels (one-way; run once, after annotation is complete).

Validates ``evaluation/datasets/blind_test_v1_labels.json`` (human provenance, every query
covered, grades 0-3, annotator ids and timestamps), then records its sha256 in
``manifest.json`` and sets ``frozen: true``. After this, any change to the label file makes
``load_split("blind_test")`` fail.

Expected label file format::

    {
      "label_source": "human",
      "annotators": [{"annotator_id": "A1", "role": "registrar office staff"}],
      "guidelines": "evaluation/judge/rubric.py relevance-rubric-v1",
      "queries": {
        "BT001": {"answerability": "unanswerable", "judgments": []},
        "BT035": {"answerability": "fully_answerable",
                  "judgments": [{"chunk_id": "HSTL-POL-001_CHUNK_002", "grade": 3,
                                 "annotator_id": "A1", "annotated_at": "2026-11-20T10:00:00Z"}]}
      }
    }

    python evaluation/datasets/freeze_blind_test.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.datasets.splits import (  # noqa: E402
    MANIFEST_PATH, ROOT, DatasetIntegrityError, load_manifest, sha256_file, validate_human_labels,
)


def main() -> int:
    manifest = load_manifest()
    info = manifest["splits"]["blind_test"]
    if info.get("frozen"):
        print("Blind test is already frozen; refusing to re-freeze. Create blind_test_v2 instead.")
        return 1
    lpath = os.path.join(ROOT, info["labels_file"])
    if not os.path.exists(lpath):
        print(f"No label file at {info['labels_file']}: nothing to freeze.")
        return 1
    with open(os.path.join(ROOT, info["queries_file"]), encoding="utf-8") as f:
        qids = [q["query_id"] for q in json.load(f)["queries"]]
    with open(lpath, encoding="utf-8") as f:
        labels = json.load(f)
    try:
        validate_human_labels(labels, qids)
    except DatasetIntegrityError as exc:
        print(f"Not frozen: {exc}")
        return 1
    info.update(frozen=True, labels_sha256=sha256_file(lpath),
                label_provenance=f"human; annotators: {[a['annotator_id'] for a in labels['annotators']]}",
                status="frozen: use only for final evaluation")
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("Frozen. labels_sha256 =", info["labels_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
