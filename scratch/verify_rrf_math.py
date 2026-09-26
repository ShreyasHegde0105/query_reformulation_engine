import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.run_evaluation import initialize_real_evaluator, load_golden_dataset
from evaluation.schema import EvaluationConfig, GoldenQuery
from reformulation.normalizer import normalize_query
from reformulation.semantic_similarity import SemanticSimilarity
from retrieval.hybrid_retriever import HybridRetriever

dataset = load_golden_dataset()
dataset_by_id = {q.query_id: q for q in dataset}
config = EvaluationConfig(use_cache=True)
evaluator = initialize_real_evaluator(config)

def analyze_query_math(query_id: str, weights=[1.0, 1.5, 2.0]):
    qitem = dataset_by_id[query_id]
    if evaluator.cached_reformulator:
        evaluator.cached_reformulator.set_current_query_id(query_id)
    
    raw = qitem.query
    norm = normalize_query(raw)
    raw_reformulations = evaluator._get_reformulations_for_query(qitem)
    candidate_queries = [norm] + raw_reformulations
    final_queries = evaluator.search_pipeline.semantic_similarity.filter_redundant(candidate_queries)
    if not final_queries:
        final_queries = [norm]

    print(f"\n{'='*80}\nDETAILED RRF MATHEMATICAL VERIFICATION FOR {query_id}\n{'='*80}")
    print(f"Raw query: {raw}")
    print(f"Normalized: {norm}")
    print(f"Relevant chunks (Ground Truth): {qitem.relevant_chunks}")
    print(f"Candidate queries ({len(candidate_queries)}):")
    for i, q in enumerate(candidate_queries):
        print(f"  [{i}] {q}")
    print(f"Active/Filtered queries ({len(final_queries)}):")
    for i, q in enumerate(final_queries):
        print(f"  Branch {i}: '{q}' (Weight mult: {'primary_query_weight' if i==0 else '1.0'})")

    # Run hybrid retrieval on each branch
    branch_results = []
    k_rrf = 60
    for i, q in enumerate(final_queries):
        res = evaluator.search_pipeline.hybrid_retriever.retrieve(
            query=q,
            vector_match_count=20,
            keyword_match_count=20,
            k=k_rrf,
            final_k=20,
        )
        branch_results.append((i, q, res))

    # Print appearances of ground truth vs top competitors across branches
    all_seen_cids = set()
    for i, q, res in branch_results:
        for ch in res:
            all_seen_cids.add(ch.chunk_id)

    print("\n--- Ground Truth Chunks in Branches ---")
    for gt_cid in qitem.relevant_chunks:
        print(f"Target GT Chunk: {gt_cid}")
        for i, q, res in branch_results:
            ranks = [r for r, ch in enumerate(res, start=1) if ch.chunk_id == gt_cid]
            if ranks:
                print(f"  Branch {i} ('{q[:35]}...'): Rank {ranks[0]} (Base RRF: 1/(60+{ranks[0]}) = {1.0/(60+ranks[0]):.6f})")
            else:
                print(f"  Branch {i} ('{q[:35]}...'): NOT RETRIEVED in top 20")

    # Run multi-query fusion for each weight and display exact math
    for w in weights:
        print(f"\n--- Multi-Query Fusion at Weight = {w} ---")
        fused = {}
        contributions_by_cid = {}
        for i, q, res in branch_results:
            branch_weight = w if i == 0 else 1.0
            for rank, ch in enumerate(res, start=1):
                contrib = branch_weight * (1.0 / (k_rrf + rank))
                fused[ch.chunk_id] = fused.get(ch.chunk_id, 0.0) + contrib
                contributions_by_cid.setdefault(ch.chunk_id, []).append((i, rank, branch_weight, contrib))

        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))

        print(f"Top 10 Ranked Chunks (Weight {w}):")
        for final_rank, (cid, total_score) in enumerate(ranked[:10], start=1):
            is_gt = "[*GROUND TRUTH*]" if cid in qitem.relevant_chunk_set else ""
            contribs_str = " + ".join([f"B{b}: {bw}*(1/{60+r})={c:.6f}" for b, r, bw, c in contributions_by_cid[cid]])
            print(f"  #{final_rank:02d} {cid:25s} Score: {total_score:.6f} {is_gt}")
            print(f"       Formula: {contribs_str} = {total_score:.6f}")

        # Show target chunk positions if beyond top 10
        for final_rank, (cid, total_score) in enumerate(ranked, start=1):
            if cid in qitem.relevant_chunk_set and final_rank > 10:
                contribs_str = " + ".join([f"B{b}: {bw}*(1/{60+r})={c:.6f}" for b, r, bw, c in contributions_by_cid[cid]])
                print(f"  #{final_rank:02d} {cid:25s} Score: {total_score:.6f} [*GROUND TRUTH*]")
                print(f"       Formula: {contribs_str} = {total_score:.6f}")

for qid in ["Q014", "Q027", "Q047", "Q015", "Q032", "Q009"]:
    analyze_query_math(qid)
