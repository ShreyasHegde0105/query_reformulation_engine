import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from evaluation.run_evaluation import load_golden_dataset, initialize_real_evaluator
from evaluation.schema import EvaluationConfig
from evaluation.evaluator import RetrievalEvaluator
from evaluation.metrics import calculate_all_metrics

dataset = load_golden_dataset()
config = EvaluationConfig(use_cache=True)
evaluator = initialize_real_evaluator(config)

print("Testing baseline vs weighted multi-query RRF on 50 queries...")

# Let's define a custom variant with original query weighting = 2.0
def custom_pipeline_weighted(query_item, cfg):
    # Search with original weight = 2.0
    top_k = max(cfg.top_k_values) if cfg.top_k_values else 20
    # Run full pipeline with weighted RRF
    raw_query = query_item.query
    normalized = evaluator.search_pipeline.semantic_similarity.filter_redundant([raw_query]) # just for test
    # Get reformulations
    reformulations = evaluator._get_reformulations_for_query(query_item)
    candidate_queries = [evaluator.search_pipeline.semantic_similarity._get_content_tokens(raw_query)] #
    # Let's run exact search_pipeline with weight 2.0 on normalized original query
    norm_q = evaluator.search_pipeline.semantic_similarity.filter_redundant([raw_query])[0]
    final_queries = evaluator.search_pipeline.semantic_similarity.filter_redundant([raw_query] + reformulations)
    
    fused = {}
    for q in final_queries:
        weight = 2.0 if q == raw_query or q == norm_q else 1.0
        results = evaluator.hybrid_retriever.retrieve(
            query=q,
            vector_match_count=cfg.vector_match_count,
            keyword_match_count=cfg.keyword_match_count,
            k=cfg.rrf_k,
            final_k=top_k,
        )
        for rank, ch in enumerate(results, start=1):
            fused[ch.chunk_id] = fused.get(ch.chunk_id, 0.0) + weight * (1.0 / (cfg.rrf_k + rank))
            
    ranked = sorted(fused.keys(), key=lambda cid: (-fused[cid], cid))
    return ranked[:top_k]

evaluator.register_variant("pipeline_weighted", custom_pipeline_weighted)

summaries = evaluator.evaluate_all_variants(dataset, ["full_pipeline", "pipeline_weighted"])

fp_m = summaries["full_pipeline"].mean_metrics
pw_m = summaries["pipeline_weighted"].mean_metrics

print("Full Pipeline Baseline:")
print(f"  R@5: {fp_m['recall@5']:.4f} | R@10: {fp_m['recall@10']:.4f} | MRR: {fp_m['mrr']:.4f} | NDCG@10: {fp_m['ndcg@10']:.4f}")

print("Full Pipeline Weighted (Original w=2.0):")
print(f"  R@5: {pw_m['recall@5']:.4f} | R@10: {pw_m['recall@10']:.4f} | MRR: {pw_m['mrr']:.4f} | NDCG@10: {pw_m['ndcg@10']:.4f}")
