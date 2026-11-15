"""Dataset split boundaries for retrieval evaluation.

``load_split(name)`` returns the labelled queries of a split in the format the metric code
expects (``query_id``, ``query``, ``category``, ``relevant_chunks``, ``graded_relevance``).

* ``development``: the tuning set.
* ``validation_contaminated``: the old held-out set, reported only under this name.
* ``blind_test``: refused (``BlindTestNotFrozenError``) unless the human labels are frozen
  and still match the hash recorded in ``manifest.json``.

Every split's query file is hash-checked against the manifest, so an edited file is detected.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(ROOT, "evaluation", "datasets", "manifest.json")
SPLITS = ("development", "validation_contaminated", "blind_test")
GRADES = (0, 1, 2, 3)


class DatasetIntegrityError(RuntimeError):
    """A query or label file does not match the hash recorded in the manifest."""


class BlindTestNotFrozenError(RuntimeError):
    """Blind-test metrics were requested before human labels were frozen."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: str = MANIFEST_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _abs(rel: str) -> str:
    return os.path.join(ROOT, rel)


def validate_human_labels(labels: Dict[str, Any], query_ids: List[str]) -> None:
    """Check a blind-test label file: human provenance, complete coverage, valid grades."""
    if labels.get("label_source") != "human":
        raise DatasetIntegrityError("blind-test labels must have label_source == 'human'")
    annotators = labels.get("annotators") or []
    if not annotators or not all(a.get("annotator_id") for a in annotators):
        raise DatasetIntegrityError("blind-test labels must list annotators with annotator_id")
    per_query = labels.get("queries") or {}
    missing = [q for q in query_ids if q not in per_query]
    if missing:
        raise DatasetIntegrityError(f"{len(missing)} blind-test queries have no labels (e.g. {missing[:3]})")
    known = {a["annotator_id"] for a in annotators}
    for qid, entry in per_query.items():
        if entry.get("answerability") not in ("fully_answerable", "partially_answerable", "unanswerable"):
            raise DatasetIntegrityError(f"{qid}: answerability must be fully_/partially_answerable or unanswerable")
        for j in entry.get("judgments", []):
            if j.get("grade") not in GRADES or not j.get("chunk_id"):
                raise DatasetIntegrityError(f"{qid}: each judgment needs chunk_id and grade 0-3")
            if j.get("annotator_id") not in known or not j.get("annotated_at"):
                raise DatasetIntegrityError(f"{qid}: each judgment needs a listed annotator_id and annotated_at")


def _labelled_from_human(labels: Dict[str, Any], queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge human judgments; with several annotators per chunk the median grade (rounded down) is used."""
    out = []
    for q in queries:
        entry = labels["queries"][q["query_id"]]
        by_chunk: Dict[str, List[int]] = {}
        for j in entry.get("judgments", []):
            by_chunk.setdefault(j["chunk_id"], []).append(int(j["grade"]))
        graded = {c: sorted(g)[(len(g) - 1) // 2] for c, g in by_chunk.items()}
        graded = {c: g for c, g in graded.items() if g > 0}
        out.append({**q, "graded_relevance": graded, "relevant_chunks": [c for c, g in graded.items() if g >= 2],
                    "answerability": entry["answerability"]})
    return out


def load_split(name: str, manifest_path: str = MANIFEST_PATH) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return ``(queries, split_info)``. Queries without relevant chunks are included (unanswerable)."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    manifest = load_manifest(manifest_path)
    info = dict(manifest["splits"][name], name=name)
    qpath = _abs(info["queries_file"])
    if sha256_file(qpath) != info["queries_sha256"]:
        raise DatasetIntegrityError(f"{info['queries_file']} does not match its manifest hash")
    with open(qpath, encoding="utf-8") as f:
        raw = json.load(f)
    queries = raw["queries"] if isinstance(raw, dict) else raw

    if name != "blind_test":
        return queries, info

    if not info.get("frozen") or not info.get("labels_sha256"):
        raise BlindTestNotFrozenError("blind-test labels are not frozen: metrics are NOT EVALUATED")
    lpath = _abs(info["labels_file"])
    if not os.path.exists(lpath) or sha256_file(lpath) != info["labels_sha256"]:
        raise DatasetIntegrityError("blind-test label file is missing or does not match its frozen hash")
    with open(lpath, encoding="utf-8") as f:
        labels = json.load(f)
    validate_human_labels(labels, [q["query_id"] for q in queries])
    return _labelled_from_human(labels, queries), info
