"""Script to analyze the 10 failure queries and score distributions for RRF and Cross-Encoder."""
import json
import os
import sys
import numpy as np

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

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

    print("===================================================================")
    print("STEP 1: 10 KNOWN FAILURE QUERIES ANALYSIS (K=50 Candidate Pool)")
    print("===================================================================")

    all_rrf_scores = []
    all_ce_scores = []

    failure_data = {}

    for qid in FAILURE_QUERY_IDS:
        q = query_map[qid]
        if evaluator.cached_reformulator:
            evaluator.cached_reformulator.set_current_query_id(qid)
        
        # Get candidate pool of 50
        candidates = evaluator.search_pipeline.search(
            query=q.query,
            max_variants=5,
            vector_match_count=20,
            keyword_match_count=20,
            k=60,
            final_k=50,
            primary_query_weight=1.0,
        )
        # Rerank with CE
        reranked = reranker.rerank(q.query, candidates)

        for c in candidates:
            all_rrf_scores.append(c.score)
        for c in reranked:
            all_ce_scores.append(c.metadata["ce_score"])

        rel_set = set(q.relevant_chunks)
        pool_ids = [c.chunk_id for c in candidates]
        reranked_ids = [c.chunk_id for c in reranked]
        present_rel = [cid for cid in q.relevant_chunks if cid in pool_ids]

        if len(present_rel) == 0:
            classification = "Type A (Candidate Generation Failure)"
        elif len(present_rel) == len(rel_set):
            classification = "Type B (Ranking Failure - All Present)"
        else:
            classification = "Type B (Ranking Failure - Partial Present)"

        failure_data[qid] = {
            "query": q.query,
            "relevant_chunks": q.relevant_chunks,
            "present_rel": present_rel,
            "classification": classification,
            "candidates": candidates,
            "reranked": reranked,
        }

        print(f"\nQuery {qid}: {q.query}")
        print(f"Classification: {classification}")
        print(f"Relevant chunks ({len(rel_set)}): {q.relevant_chunks}")
        print(f"Present in candidate pool ({len(present_rel)}): {present_rel}")

        for r_id in q.relevant_chunks:
            rrf_pos = pool_ids.index(r_id) + 1 if r_id in pool_ids else "ABSENT"
            ce_pos = reranked_ids.index(r_id) + 1 if r_id in reranked_ids else "ABSENT"
            ce_score = None
            if isinstance(ce_pos, int):
                ce_score = reranked[ce_pos - 1].metadata.get("ce_score")
            print(f"  Target {r_id}: RRF Rank={rrf_pos} -> CE Rank={ce_pos}, CE Score={ce_score}")

    print("\n===================================================================")
    print("STEP 2: SCORE DISTRIBUTION ANALYSIS (across 10 failure queries, 500 candidates)")
    print("===================================================================")
    rrf_arr = np.array(all_rrf_scores)
    ce_arr = np.array(all_ce_scores)

    print(f"RRF Scores: count={len(rrf_arr)}, min={rrf_arr.min():.5f}, max={rrf_arr.max():.5f}, mean={rrf_arr.mean():.5f}, std={rrf_arr.std():.5f}")
    print(f"  Percentiles [p25, p50, p75, p95]: {np.percentile(rrf_arr, [25, 50, 75, 95])}")

    print(f"CE Scores (logits): count={len(ce_arr)}, min={ce_arr.min():.4f}, max={ce_arr.max():.4f}, mean={ce_arr.mean():.4f}, std={ce_arr.std():.4f}")
    print(f"  Percentiles [p25, p50, p75, p95]: {np.percentile(ce_arr, [25, 50, 75, 95])}")

    # Inspect Q014 specifically
    print("\n===================================================================")
    print("STEP 3: SPECIAL ANALYSIS — Q014")
    print("===================================================================")
    q014 = query_map["Q014"]
    print(f"Query: {q014.query}")
    print(f"Relevant chunks: {q014.relevant_chunks}")
    # Check candidate pool at K=20, 30, 50
    for k_val in [20, 30, 50]:
        cands_k = evaluator.search_pipeline.search(
            query=q014.query,
            max_variants=5,
            vector_match_count=20,
            keyword_match_count=20,
            k=60,
            final_k=k_val,
            primary_query_weight=1.0,
        )
        c_ids = [c.chunk_id for c in cands_k]
        found = [r for r in q014.relevant_chunks if r in c_ids]
        print(f"  K={k_val}: found {len(found)}/{len(q014.relevant_chunks)} -> {found}")

    # Inspect Q024 specifically
    print("\n===================================================================")
    print("STEP 4: SPECIAL ANALYSIS — Q024")
    print("===================================================================")
    q024_data = failure_data["Q024"]
    print("Top 10 RRF candidates:")
    for idx, c in enumerate(q024_data["candidates"][:10], 1):
        is_rel = "[REL]" if c.chunk_id in q024_data["relevant_chunks"] else "     "
        print(f"  {idx:2d}. {is_rel} {c.chunk_id} (RRF: {c.score:.5f}) - {c.title}")
    print("\nTop 10 CE reranked candidates:")
    for idx, c in enumerate(q024_data["reranked"][:10], 1):
        is_rel = "[REL]" if c.chunk_id in q024_data["relevant_chunks"] else "     "
        print(f"  {idx:2d}. {is_rel} {c.chunk_id} (CE: {c.metadata['ce_score']:.4f}, RRF: {c.score:.5f}) - {c.title}")

    # Inspect Q004 specifically
    print("\n===================================================================")
    print("STEP 5: SPECIAL ANALYSIS — Q004")
    print("===================================================================")
    q004_data = failure_data["Q004"]
    print("Top 5 RRF candidates:")
    for idx, c in enumerate(q004_data["candidates"][:5], 1):
        print(f"  {idx:2d}. {c.chunk_id} (RRF: {c.score:.5f}) - {c.title}")
    print("\nTop 5 CE candidates:")
    for idx, c in enumerate(q004_data["reranked"][:5], 1):
        print(f"  {idx:2d}. {c.chunk_id} (CE: {c.metadata['ce_score']:.4f}) - {c.title}")

    # Inspect Q027 specifically
    print("\n===================================================================")
    print("STEP 6: SPECIAL ANALYSIS — Q027")
    print("===================================================================")
    q027_data = failure_data["Q027"]
    print("Top 10 RRF candidates:")
    for idx, c in enumerate(q027_data["candidates"][:10], 1):
        is_rel = "[REL]" if c.chunk_id in q027_data["relevant_chunks"] else "     "
        print(f"  {idx:2d}. {is_rel} {c.chunk_id} (RRF: {c.score:.5f}) - {c.title}")
    print("\nTop 10 CE candidates:")
    for idx, c in enumerate(q027_data["reranked"][:10], 1):
        is_rel = "[REL]" if c.chunk_id in q027_data["relevant_chunks"] else "     "
        print(f"  {idx:2d}. {is_rel} {c.chunk_id} (CE: {c.metadata['ce_score']:.4f}) - {c.title}")

    # Inspect Q047 specifically
    print("\n===================================================================")
    print("STEP 7: SPECIAL ANALYSIS — Q047")
    print("===================================================================")
    q047_data = failure_data["Q047"]
    print("Top 10 RRF candidates:")
    for idx, c in enumerate(q047_data["candidates"][:10], 1):
        is_rel = "[REL]" if c.chunk_id in q047_data["relevant_chunks"] else "     "
        print(f"  {idx:2d}. {is_rel} {c.chunk_id} (RRF: {c.score:.5f}) - {c.title}")
    print("\nTop 10 CE candidates:")
    for idx, c in enumerate(q047_data["reranked"][:10], 1):
        is_rel = "[REL]" if c.chunk_id in q047_data["relevant_chunks"] else "     "
        print(f"  {idx:2d}. {is_rel} {c.chunk_id} (CE: {c.metadata['ce_score']:.4f}) - {c.title}")

if __name__ == "__main__":
    main()
