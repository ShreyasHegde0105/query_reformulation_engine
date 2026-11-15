"""Build the human annotation pool for the blind test (TREC-style pooling; NO grades are produced).

For every blind-test query, the top-10 chunks of configurations A-D (same settings as
run_ad_benchmark.py, candidate limit 50) are pooled, de-duplicated and written in a
hash-based order (so annotators cannot tell which system or rank produced a chunk) to:

* blind_test_v1_annotation_pool.csv   one row per (query, chunk): grade / annotator_id / annotated_at left BLANK
* blind_test_v1_answerability.csv     one row per query: answerability / annotator_id / annotated_at left BLANK

Annotators grade with the rubric in evaluation/judge/rubric.py (0-3). Chunks outside the pool
can be added as extra rows. Then run labels_from_csv.py and freeze_blind_test.py.

    python evaluation/datasets/build_annotation_pool.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from evaluation.run_ad_benchmark import CHUNKS_PATH, REWRITE_CACHES, build  # noqa: E402

HERE = os.path.join(ROOT, "evaluation", "datasets")


def main() -> int:
    data = json.load(open(os.path.join(HERE, "blind_test_v1.json"), encoding="utf-8"))
    queries = data["queries"]
    cache_path = REWRITE_CACHES["blind_test"]
    if not os.path.exists(cache_path):
        print(f"Missing rewrite cache {cache_path}; run build_reformulation_cache.py first.")
        return 2
    rdata = json.load(open(cache_path, encoding="utf-8"))
    rewrites = rdata.get("reformulations", rdata)
    chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8") if l.strip()]
    by_id = {c["chunk_id"]: c for c in chunks}
    pipelines, cached, _ = build(chunks, rewrites, "cross-encoder/ms-marco-MiniLM-L-6-v2", 50)

    pool_rows, ans_rows = [], []
    for q in queries:
        cached.query_id = q["query_id"]
        pooled = set()
        for name, p in pipelines.items():
            res = p.search_detailed(q["query"], max_variants=5, vector_match_count=50, keyword_match_count=50,
                                    k=60, final_k=10, candidate_k=50)
            pooled.update(c.chunk_id for c in res.results)
        order = sorted(pooled, key=lambda cid: hashlib.sha256(f"{q['query_id']}|{cid}".encode()).hexdigest())
        for cid in order:
            c = by_id[cid]
            pool_rows.append({"query_id": q["query_id"], "query": q["query"], "category": q["category"],
                              "chunk_id": cid, "document_id": c["document_id"], "title": c["title"],
                              "section": c.get("section") or "", "text": c["text"],
                              "grade": "", "annotator_id": "", "annotated_at": "", "notes": ""})
        ans_rows.append({"query_id": q["query_id"], "query": q["query"], "category": q["category"],
                         "proposed_answerability": q["proposed_answerability"], "answerability": "",
                         "annotator_id": "", "annotated_at": "", "notes": ""})
        print(q["query_id"], len(order), "pooled chunks", flush=True)

    for name, rows in (("blind_test_v1_annotation_pool.csv", pool_rows), ("blind_test_v1_answerability.csv", ans_rows)):
        with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(pool_rows)} (query, chunk) rows for {len(queries)} queries; grades are blank by design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
