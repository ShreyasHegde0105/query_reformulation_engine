"""Phase 4 — Step 1: Candidate Pool Analysis

Runs the SearchPipeline at three candidate pool sizes (K=20, K=30, K=50)
across all 50 golden queries and measures:
  - Candidate Recall@K  (fraction of relevant chunks present in the pool)
  - Per-query classification:
      ALL_PRESENT  — all relevant chunks are in the pool
      SOME_PRESENT — at least one (but not all) relevant chunks in pool
      NONE_PRESENT — no relevant chunks in pool at all

Each query failure in Phase 3 is then classified as:
  Type A — no relevant chunk in pool even at K=50 (CE cannot help)
  Type B — at least one relevant chunk present at some K (CE may help)

Outputs:
  evaluation/results/phase4_candidate_pool_analysis.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Set

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.run_evaluation import initialize_real_evaluator
from evaluation.schema import EvaluationConfig, GoldenQuery

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLDEN_PATH = os.path.join(PROJECT_ROOT, "evaluation", "golden_queries.json")
CACHE_PATH = os.path.join(PROJECT_ROOT, "evaluation", "cache", "reformulations.json")
OUTPUT_MD = os.path.join(PROJECT_ROOT, "evaluation", "results", "phase4_candidate_pool_analysis.md")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "evaluation", "results", "phase4_candidate_pool_raw.json")

CANDIDATE_K_VALUES = [20, 30, 50]

# Phase 3 identified failure queries
FAILURE_QUERY_IDS = {"Q004", "Q005", "Q008", "Q014", "Q016", "Q017", "Q020", "Q024", "Q027", "Q047"}

# Phase-3 confirmed Type A failures (candidate generation)
KNOWN_TYPE_A = {"Q014"}  # Target absent from ALL 6 branches at K=20

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_golden_queries() -> List[GoldenQuery]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    queries = []
    for item in raw:
        q = GoldenQuery(**item)
        queries.append(q)
    return queries


def load_reformulation_cache() -> Dict[str, List[str]]:
    if not os.path.exists(CACHE_PATH):
        print(f"WARNING: reformulation cache not found at {CACHE_PATH}")
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Candidate pool retrieval
# ---------------------------------------------------------------------------

def get_candidate_pool(
    evaluator,
    query_item: GoldenQuery,
    final_k: int,
) -> List[str]:
    """Return the top-final_k chunk IDs from the full pipeline for a query."""
    # Set cached reformulator context
    if evaluator.cached_reformulator:
        evaluator.cached_reformulator.set_current_query_id(query_item.query_id)

    chunks = evaluator.search_pipeline.search(
        query=query_item.query,
        max_variants=5,
        vector_match_count=20,
        keyword_match_count=20,
        k=60,
        final_k=final_k,
        primary_query_weight=1.0,  # standard RRF — Phase 2 / Phase 3 w=1.0 baseline
    )
    return [c.chunk_id for c in chunks]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_presence(
    candidate_ids: List[str],
    relevant_set: Set[str],
) -> str:
    """Classify candidate pool coverage."""
    present = relevant_set.intersection(set(candidate_ids))
    if len(present) == len(relevant_set):
        return "ALL_PRESENT"
    elif len(present) > 0:
        return "SOME_PRESENT"
    else:
        return "NONE_PRESENT"


def candidate_recall(candidate_ids: List[str], relevant_set: Set[str]) -> float:
    """Fraction of relevant chunks present in the candidate pool."""
    if not relevant_set:
        return 0.0
    present = relevant_set.intersection(set(candidate_ids))
    return len(present) / len(relevant_set)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_candidate_pool_analysis():
    print("=" * 70)
    print("Phase 4 — Candidate Pool Analysis")
    print("=" * 70)
    print()

    # Initialize evaluator
    print("Initializing evaluator...")
    config = EvaluationConfig(use_cache=True)
    evaluator = initialize_real_evaluator(config)
    print("Evaluator ready.\n")

    # Load dataset
    queries = load_golden_queries()
    print(f"Loaded {len(queries)} golden queries.\n")

    # Results container
    results: Dict[str, Dict] = {}
    agg_recall: Dict[int, float] = {k: 0.0 for k in CANDIDATE_K_VALUES}
    class_counts: Dict[int, Dict[str, int]] = {
        k: {"ALL_PRESENT": 0, "SOME_PRESENT": 0, "NONE_PRESENT": 0}
        for k in CANDIDATE_K_VALUES
    }

    print(f"{'Query':8} | {'Rel':4} | {'K=20 Recall':11} | {'K=30 Recall':11} | {'K=50 Recall':11} | {'K=20 Status':14} | {'K=50 Status':14}")
    print("-" * 100)

    for query_item in queries:
        qid = query_item.query_id
        relevant_set = set(query_item.relevant_chunks)
        n_rel = len(relevant_set)

        row: Dict = {
            "query_id": qid,
            "query": query_item.query,
            "n_relevant": n_rel,
            "relevant_chunks": list(relevant_set),
            "candidate_pools": {},
        }

        recalls: Dict[int, float] = {}
        statuses: Dict[int, str] = {}

        for k in CANDIDATE_K_VALUES:
            t0 = time.perf_counter()
            pool = get_candidate_pool(evaluator, query_item, final_k=k)
            elapsed = time.perf_counter() - t0

            recall = candidate_recall(pool, relevant_set)
            status = classify_presence(pool, relevant_set)

            recalls[k] = recall
            statuses[k] = status
            agg_recall[k] += recall
            class_counts[k][status] += 1

            # Record which relevant chunks are in pool
            present_ids = relevant_set.intersection(set(pool))
            absent_ids = relevant_set - set(pool)

            row["candidate_pools"][k] = {
                "pool_size": len(pool),
                "candidate_recall": recall,
                "status": status,
                "present_relevant": sorted(present_ids),
                "absent_relevant": sorted(absent_ids),
                "latency_s": elapsed,
            }

        results[qid] = row

        # Failure classification
        failure_marker = " << FAILURE" if qid in FAILURE_QUERY_IDS else ""
        print(
            f"{qid:8} | {n_rel:4} | "
            f"{recalls[20]:.3f}{'':7} | "
            f"{recalls[30]:.3f}{'':7} | "
            f"{recalls[50]:.3f}{'':7} | "
            f"{statuses[20]:14} | "
            f"{statuses[50]:14}"
            f"{failure_marker}"
        )

    n = len(queries)
    print("\n" + "=" * 100)
    print(f"{'AGGREGATE':8} | {'':4} | {agg_recall[20]/n:.3f}{'':7} | {agg_recall[30]/n:.3f}{'':7} | {agg_recall[50]/n:.3f}{'':7} |")
    print("=" * 100)

    # ---------------------------------------------------------------------------
    # Failure-query deep dive
    # ---------------------------------------------------------------------------
    print("\n--- Failure Query Classification ---\n")
    type_a_queries = []
    type_b_queries = []

    for qid in sorted(FAILURE_QUERY_IDS):
        if qid not in results:
            continue
        row = results[qid]
        pool50 = row["candidate_pools"][50]

        if pool50["status"] == "NONE_PRESENT":
            failure_type = "Type A (candidate gen failure — CE cannot help)"
            type_a_queries.append(qid)
        else:
            failure_type = "Type B (ranking failure — CE may help)"
            type_b_queries.append(qid)

        absent_at_50 = pool50["absent_relevant"]
        print(f"  {qid}: {failure_type}")
        print(f"    K=20 recall={row['candidate_pools'][20]['candidate_recall']:.3f}  "
              f"K=30={row['candidate_pools'][30]['candidate_recall']:.3f}  "
              f"K=50={row['candidate_pools'][50]['candidate_recall']:.3f}")
        if absent_at_50:
            print(f"    Absent at K=50: {absent_at_50}")
        print()

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "aggregate": {
                    "mean_candidate_recall": {k: agg_recall[k] / n for k in CANDIDATE_K_VALUES},
                    "classification_counts": class_counts,
                },
                "type_a_queries": type_a_queries,
                "type_b_queries": type_b_queries,
                "per_query": results,
            },
            f,
            indent=2,
        )
    print(f"Saved raw JSON: {OUTPUT_JSON}")

    # ---------------------------------------------------------------------------
    # Generate markdown report
    # ---------------------------------------------------------------------------
    _write_markdown_report(
        results=results,
        agg_recall=agg_recall,
        class_counts=class_counts,
        type_a_queries=type_a_queries,
        type_b_queries=type_b_queries,
        n=n,
    )
    print(f"Saved markdown: {OUTPUT_MD}")

    return results, type_a_queries, type_b_queries


def _write_markdown_report(
    results: Dict,
    agg_recall: Dict[int, float],
    class_counts: Dict[int, Dict[str, int]],
    type_a_queries: List[str],
    type_b_queries: List[str],
    n: int,
) -> None:
    lines = []
    lines.append("# Phase 4 — Candidate Pool Analysis\n")
    lines.append("> **Baseline**: Phase-2 Full Pipeline (weight=1.0, K=20)  \n")
    lines.append("> **Question**: How many relevant chunks reach the candidate pool at K=20, K=30, K=50?\n\n")

    lines.append("## 1. Aggregate Candidate Recall\n")
    lines.append("| Pool Size | Mean Candidate Recall | ALL_PRESENT | SOME_PRESENT | NONE_PRESENT |")
    lines.append("|-----------|----------------------|-------------|--------------|--------------|")
    for k in [20, 30, 50]:
        cc = class_counts[k]
        lines.append(
            f"| K={k:<3}     | {agg_recall[k]/n:.4f}               "
            f"| {cc['ALL_PRESENT']:11} | {cc['SOME_PRESENT']:12} | {cc['NONE_PRESENT']:12} |"
        )
    lines.append("")

    lines.append("## 2. Failure Query Analysis\n")
    lines.append(f"> Phase 3 identified **{len(FAILURE_QUERY_IDS)} structural failure queries**.\n")
    lines.append(
        f"> Cross-encoder can address **Type B only** "
        f"(relevant chunk present but ranked too low).\n"
    )

    lines.append("### Type A — Candidate Generation Failures (CE cannot help)\n")
    if type_a_queries:
        for qid in sorted(type_a_queries):
            row = results[qid]
            lines.append(f"- **{qid}**: `{row['query'][:80]}`")
            for k in [20, 30, 50]:
                s = row["candidate_pools"][k]
                lines.append(f"  - K={k}: recall={s['candidate_recall']:.3f} ({s['status']})")
                if s["absent_relevant"]:
                    lines.append(f"  - Absent chunks at K={k}: `{s['absent_relevant']}`")
    else:
        lines.append("*(None — all failure queries have at least one relevant chunk in pool at K=50)*\n")

    lines.append("\n### Type B — Ranking Failures (CE may help)\n")
    if type_b_queries:
        for qid in sorted(type_b_queries):
            row = results[qid]
            lines.append(f"- **{qid}**: `{row['query'][:80]}`")
            for k in [20, 30, 50]:
                s = row["candidate_pools"][k]
                lines.append(f"  - K={k}: recall={s['candidate_recall']:.3f} ({s['status']})")
    else:
        lines.append("*(None)*\n")

    lines.append("\n## 3. Per-Query Detail Table\n")
    lines.append("| Query | Rel | K=20 Recall | K=30 Recall | K=50 Recall | K=20 Status | K=50 Status |")
    lines.append("|-------|-----|-------------|-------------|-------------|-------------|-------------|")
    for qid, row in sorted(results.items()):
        p20 = row["candidate_pools"][20]
        p30 = row["candidate_pools"][30]
        p50 = row["candidate_pools"][50]
        failure = " ⚠" if qid in FAILURE_QUERY_IDS else ""
        lines.append(
            f"| {qid}{failure} | {row['n_relevant']} "
            f"| {p20['candidate_recall']:.3f} "
            f"| {p30['candidate_recall']:.3f} "
            f"| {p50['candidate_recall']:.3f} "
            f"| {p20['status']} "
            f"| {p50['status']} |"
        )

    lines.append("\n## 4. Key Findings\n")
    lines.append(
        f"- **Mean candidate recall at K=50**: {agg_recall[50]/n:.4f} "
        f"(vs K=20: {agg_recall[20]/n:.4f})\n"
    )
    lines.append(
        f"- **Type A queries** (CE powerless): {len(type_a_queries)} — {sorted(type_a_queries)}\n"
    )
    lines.append(
        f"- **Type B queries** (CE may help): {len(type_b_queries)} — {sorted(type_b_queries)}\n"
    )
    lines.append(
        "- Increasing candidate pool from K=20 to K=50 may convert some Type A → Type B.\n"
    )

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_candidate_pool_analysis()
