import json
import logging
import os
import sys
from typing import Any, Dict, List

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import calculate_all_metrics
from evaluation.run_evaluation import initialize_real_evaluator, load_golden_dataset
from evaluation.schema import EvaluationConfig, GoldenQuery, QueryResult
from reformulation.search_pipeline import SearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rrf_audit")


def run_weight_experiment(
    evaluator,
    dataset: List[GoldenQuery],
    weight: float,
    k_rrf: int = 60,
    top_k: int = 20,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"Multi-query RRF configuration:")
    print(f"primary_query_weight = {weight}")
    print(f"rrf_k = {k_rrf}")
    print(f"vector_match_count = 20")
    print(f"keyword_match_count = 20")
    print(f"final_k = {top_k}")
    print("=" * 80)

    results: List[QueryResult] = []
    per_query_details: Dict[str, Any] = {}

    for query_item in dataset:
        if evaluator.cached_reformulator:
            evaluator.cached_reformulator.set_current_query_id(query_item.query_id)

        # Call search pipeline with explicit primary_query_weight
        chunks = evaluator.search_pipeline.search(
            query=query_item.query,
            max_variants=evaluator.config.max_reformulation_variants,
            vector_match_count=evaluator.config.vector_match_count,
            keyword_match_count=evaluator.config.keyword_match_count,
            k=k_rrf,
            final_k=top_k,
            primary_query_weight=weight,
        )

        retrieved_chunk_ids = [c.chunk_id for c in chunks]
        reformulations = evaluator._get_reformulations_for_query(query_item)

        metrics = calculate_all_metrics(
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=query_item.relevant_chunks,
            graded_relevance=query_item.graded_relevance,
            k_list=[5, 10],
        )

        qres = QueryResult(
            query_id=query_item.query_id,
            query=query_item.query,
            category=query_item.category,
            variant=f"full_pipeline_w{weight}",
            retrieved_chunks=retrieved_chunk_ids,
            relevant_chunks=query_item.relevant_chunks,
            metrics=metrics,
            reformulations=reformulations,
        )
        results.append(qres)

        # Record top chunks and scores
        per_query_details[query_item.query_id] = {
            "query_id": query_item.query_id,
            "query": query_item.query,
            "category": query_item.category,
            "relevant_chunks": query_item.relevant_chunks,
            "metrics": metrics,
            "retrieved_chunks": [
                {
                    "rank": idx + 1,
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                    "document_id": c.document_id,
                    "is_relevant": c.chunk_id in query_item.relevant_chunk_set,
                }
                for idx, c in enumerate(chunks)
            ],
        }

    # Aggregate metrics
    metric_sums: Dict[str, float] = {}
    for r in results:
        for k, v in r.metrics.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v

    total = len(results)
    mean_metrics = {k: (v / total if total > 0 else 0.0) for k, v in metric_sums.items()}

    # Category breakdown
    cat_groups: Dict[str, List[QueryResult]] = {}
    for r in results:
        cat_groups.setdefault(r.category, []).append(r)

    category_metrics: Dict[str, Dict[str, float]] = {}
    for cat, cat_res in cat_groups.items():
        c_sums: Dict[str, float] = {}
        for r in cat_res:
            for k, v in r.metrics.items():
                c_sums[k] = c_sums.get(k, 0.0) + v
        c_tot = len(cat_res)
        category_metrics[cat] = {k: (v / c_tot if c_tot > 0 else 0.0) for k, v in c_sums.items()}

    return {
        "weight": weight,
        "mean_metrics": mean_metrics,
        "category_metrics": category_metrics,
        "per_query_results": [r.to_dict() for r in results],
        "per_query_details": per_query_details,
    }


def main():
    dataset = load_golden_dataset()
    config = EvaluationConfig(use_cache=True)
    evaluator = initialize_real_evaluator(config)

    print(f"Loaded {len(dataset)} golden queries.")

    exp_1_0 = run_weight_experiment(evaluator, dataset, weight=1.0)
    exp_1_5 = run_weight_experiment(evaluator, dataset, weight=1.5)
    exp_2_0 = run_weight_experiment(evaluator, dataset, weight=2.0)

    # Save to json
    output_payload = {
        "experiments": {
            "weight_1.0": exp_1_0,
            "weight_1.5": exp_1_5,
            "weight_2.0": exp_2_0,
        }
    }

    out_file = os.path.join(os.path.dirname(__file__), "..", "evaluation", "results", "rrf_weight_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved experiment results to {out_file}")

    # Print summary comparison table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<15} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<10} | {'NDCG@5':<10} | {'NDCG@10':<10}")
    print("-" * 80)
    for name, exp in [("Weight 1.0", exp_1_0), ("Weight 1.5", exp_1_5), ("Weight 2.0", exp_2_0)]:
        m = exp["mean_metrics"]
        print(f"{name:<15} | {m['recall@5']:<10.4f} | {m['recall@10']:<10.4f} | {m['mrr']:<10.4f} | {m['ndcg@5']:<10.4f} | {m['ndcg@10']:<10.4f}")


if __name__ == "__main__":
    main()
