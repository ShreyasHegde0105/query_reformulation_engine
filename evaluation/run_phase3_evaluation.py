from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.evaluator import RetrievalEvaluator
from evaluation.run_evaluation import (
    generate_markdown_report,
    initialize_real_evaluator,
    load_golden_dataset,
    save_json_results,
)
from evaluation.schema import EvaluationConfig, EvaluationSummary, GoldenQuery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluation.phase3_runner")


def compare_phase2_vs_phase3(
    phase2_json_path: str,
    phase3_json_path: str,
    output_regressions_path: str,
) -> Dict[str, Any]:
    """Compare Phase 2 and Phase 3 evaluation results and generate regressions report."""
    if not os.path.exists(phase2_json_path) or not os.path.exists(phase3_json_path):
        logger.warning("Comparison skipped: missing results file(s).")
        return {}

    with open(phase2_json_path, "r", encoding="utf-8") as f:
        p2_data = json.load(f)
    with open(phase3_json_path, "r", encoding="utf-8") as f:
        p3_data = json.load(f)

    p2_full = p2_data["summaries"]["full_pipeline"]["per_query_results"]
    p3_full = p3_data["summaries"]["full_pipeline"]["per_query_results"]

    p2_by_qid = {r["query_id"]: r for r in p2_full}
    p3_by_qid = {r["query_id"]: r for r in p3_full}

    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []

    for qid, p3_res in p3_by_qid.items():
        p2_res = p2_by_qid.get(qid)
        if not p2_res:
            continue

        p2_rec10 = p2_res["metrics"].get("recall@10", 0.0)
        p3_rec10 = p3_res["metrics"].get("recall@10", 0.0)
        p2_mrr = p2_res["metrics"].get("mrr", 0.0)
        p3_mrr = p3_res["metrics"].get("mrr", 0.0)

        item_diff = {
            "query_id": qid,
            "query": p3_res["query"],
            "category": p3_res["category"],
            "phase2_recall10": p2_rec10,
            "phase3_recall10": p3_rec10,
            "phase2_mrr": p2_mrr,
            "phase3_mrr": p3_mrr,
            "delta_recall10": p3_rec10 - p2_rec10,
            "delta_mrr": p3_mrr - p2_mrr,
        }

        if p3_mrr < p2_mrr or p3_rec10 < p2_rec10:
            regressions.append(item_diff)
        elif p3_mrr > p2_mrr or p3_rec10 > p2_rec10:
            improvements.append(item_diff)
        else:
            unchanged.append(item_diff)

    # Generate regressions markdown
    lines = [
        "# Phase 3 Regressions and Query Performance Analysis",
        "",
        f"**Date:** {datetime.now(timezone.utc).isoformat()}",
        f"**Total Queries Evaluated:** {len(p3_by_qid)}",
        f"**Improved Queries:** {len(improvements)}",
        f"**Regressed Queries:** {len(regressions)}",
        f"**Unchanged Queries:** {len(unchanged)}",
        "",
        "## 1. Regressed Queries",
        "",
    ]

    if not regressions:
        lines.append("*No regressions detected across the 50-query benchmark.*")
    else:
        lines.extend([
            "| Query ID | Category | Query Text | Phase 2 MRR | Phase 3 MRR | Phase 2 Rec@10 | Phase 3 Rec@10 | Probable Reason |",
            "|:---|:---|:---|---:|---:|---:|---:|:---|",
        ])
        for reg in regressions:
            lines.append(
                f"| {reg['query_id']} | {reg['category']} | {reg['query']} | {reg['phase2_mrr']:.4f} | {reg['phase3_mrr']:.4f} | {reg['phase2_recall10']:.4f} | {reg['phase3_recall10']:.4f} | Primary query weighting shift |"
            )

    lines.extend([
        "",
        "## 2. Improved Queries",
        "",
        "| Query ID | Category | Query Text | Phase 2 MRR | Phase 3 MRR | Phase 2 Rec@10 | Phase 3 Rec@10 |",
        "|:---|:---|:---|---:|---:|---:|---:|",
    ])
    for imp in improvements:
        lines.append(
            f"| {imp['query_id']} | {imp['category']} | {imp['query']} | {imp['phase2_mrr']:.4f} | {imp['phase3_mrr']:.4f} | {imp['phase2_recall10']:.4f} | {imp['phase3_recall10']:.4f} |"
        )

    with open(output_regressions_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved regressions report to: %s", output_regressions_path)
    return {
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
    }


def main() -> None:
    dataset = load_golden_dataset()
    config = EvaluationConfig(use_cache=True, primary_query_weight=1.5)
    evaluator = initialize_real_evaluator(config)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(dataset),
        "corpus_chunks": 696,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "top_k_values": config.top_k_values,
        "rrf_k": config.rrf_k,
        "phase": 3,
        "primary_query_weight": 1.5,
    }

    variants = ["baseline_vector", "baseline_keyword", "baseline_hybrid", "reformulation_hybrid", "full_pipeline"]
    summaries = evaluator.evaluate_all_variants(dataset, variants, metadata=metadata)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    json_path = os.path.join(results_dir, "phase3_evaluation_results.json")
    md_path = os.path.join(results_dir, "phase3_evaluation_report.md")
    reg_path = os.path.join(results_dir, "phase3_regressions.md")

    save_json_results(summaries, json_path, metadata=metadata)
    generate_markdown_report(summaries, md_path, metadata=metadata)

    phase2_json_path = os.path.join(results_dir, "evaluation_results.json")
    compare_phase2_vs_phase3(phase2_json_path, json_path, reg_path)

    print("PHASE 3 EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
