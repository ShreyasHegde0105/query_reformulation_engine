"""Script to run the 10 failure queries and inspect score distributions."""
import json
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.run_evaluation import initialize_real_evaluator
from evaluation.schema import EvaluationConfig, GoldenQuery
from reformulation.cross_encoder import CrossEncoderReranker
from evaluation.metrics import calculate_all_metrics

FAILURE_QUERY_IDS = ["Q004", "Q005", "Q008", "Q014", "Q016", "Q017", "Q020", "Q024", "Q027", "Q047"]

def main():
    with open("evaluation/golden_queries.json", "r", encoding="utf-8") as f:
        all_queries = [GoldenQuery(**item) for item in json.load(f)]
    query_map = {q.query_id: q for q in all_queries}

    config = EvaluationConfig(use_cache=True)
    evaluator = initialize_real_evaluator(config)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32, enabled=True)
    reranker._load_model()

    print("Running 10 Failure Queries at K=50 pool...")
    for qid in FAILURE_QUERY_IDS:
        q = query_map[qid]
        if evaluator.cached_reformulator:
            evaluator.cached_reformulator.set_current_query_id(qid)
        candidates = evaluator.search_pipeline.search(
            query=q.query,
            max_variants=5,
            vector_match_count=20,
            keyword_match_count=20,
            k=60,
            final_k=50,
            primary_query_weight=1.0,
        )
        reranked = reranker.rerank(q.query, candidates)
        
        rel_set = set(q.relevant_chunks)
        pool_ids = [c.chunk_id for c in candidates]
        present_rel = [cid for cid in q.relevant_chunks if cid in pool_ids]
        
        print(f"\n=======================================================")
        print(f"Query {qid}: {q.query}")
        print(f"Relevant chunks ({len(rel_set)}): {q.relevant_chunks}")
        print(f"Present in K=50 pool ({len(present_rel)}): {present_rel}")
        ftype = "Type A (Candidate Gen Failure)" if len(present_rel) == 0 else ("Type B (Ranking Failure)" if len(present_rel) == len(rel_set) else "Type B partial")
        print(f"Classification: {ftype}")
        
        # Show positions of relevant chunks
        for r_id in q.relevant_chunks:
            rrf_pos = pool_ids.index(r_id) + 1 if r_id in pool_ids else None
            ce_pos = [c.chunk_id for c in reranked].index(r_id) + 1 if r_id in [c.chunk_id for c in reranked] else None
            ce_score = None
            if ce_pos:
                ce_score = reranked[ce_pos - 1].metadata.get("ce_score")
            print(f"  Target {r_id}: RRF Rank={rrf_pos}, CE Rank={ce_pos}, CE Score={ce_score}")

        # Top 5 by RRF vs Top 5 by CE
        print("\n  Top 5 RRF Candidates:")
        for idx, c in enumerate(candidates[:5], 1):
            is_rel = "✓ REL" if c.chunk_id in rel_set else "  irr"
            print(f"    {idx}. [{is_rel}] {c.chunk_id} (RRF: {c.score:.5f}) - {c.title[:40]}")

        print("\n  Top 5 CE Reranked Candidates:")
        for idx, c in enumerate(reranked[:5], 1):
            is_rel = "✓ REL" if c.chunk_id in rel_set else "  irr"
            ce_s = c.metadata.get("ce_score", 0.0)
            print(f"    {idx}. [{is_rel}] {c.chunk_id} (CE: {ce_s:.4f}, RRF: {c.score:.5f}) - {c.title[:40]}")

if __name__ == "__main__":
    main()
