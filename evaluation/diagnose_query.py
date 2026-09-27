from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.schema import GoldenQuery
from evaluation.run_evaluation import load_golden_dataset
from reformulation.normalizer import normalize_query
from reformulation.semantic_similarity import SemanticSimilarity
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.models import RetrievedChunk

logger = logging.getLogger("evaluation.diagnose")


def get_query_reformulations(query_id: str, default_query: str) -> List[str]:
    """Load cached reformulations for a query_id."""
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "reformulations.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
            if query_id in cache:
                return cache[query_id]
    return []


def trace_semantic_filtering(
    candidate_queries: List[str],
    similarity_engine: SemanticSimilarity,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Trace semantic filtering decisions step-by-step."""
    accepted_queries = similarity_engine.filter_redundant(candidate_queries)
    accepted_set = set(accepted_queries)

    pruning_log: List[Dict[str, Any]] = []
    for cand in candidate_queries:
        if cand not in accepted_set:
            cand_toks = similarity_engine._get_content_tokens(cand)
            rejection_reason = "Pruned as redundant"
            matched_rep = None
            max_sim = 0.0

            for acc in accepted_queries:
                sim = similarity_engine.similarity(cand, acc)
                if sim > max_sim:
                    max_sim = sim
                    matched_rep = acc
                acc_toks = similarity_engine._get_content_tokens(acc)
                novel_toks = cand_toks - acc_toks
                if sim >= similarity_engine.threshold and len(novel_toks) == 0:
                    rejection_reason = f"Redundant with '{acc}' (sim={sim:.4f}, novel_tokens={list(novel_toks)})"
                    break

            pruning_log.append({
                "candidate": cand,
                "status": "REMOVED",
                "reason": rejection_reason,
                "highest_similarity": max_sim,
                "most_similar_rep": matched_rep,
            })
        else:
            pruning_log.append({
                "candidate": cand,
                "status": "ACCEPTED",
                "reason": "Retained as representative / novel terminology",
            })

    return accepted_queries, pruning_log


def diagnose_query(
    query_id: str,
    dataset_path: Optional[str] = None,
    k_rrf: int = 60,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Trace a query end-to-end through every component and report failure diagnostics."""
    dataset = load_golden_dataset(dataset_path)
    target_item: Optional[GoldenQuery] = None
    for q in dataset:
        if q.query_id == query_id:
            target_item = q
            break

    if target_item is None:
        raise ValueError(f"Query ID '{query_id}' not found in golden dataset.")

    # 1. Pipeline stages
    raw_query = target_item.query
    normalized_query = normalize_query(raw_query)
    raw_reformulations = get_query_reformulations(query_id, raw_query)
    
    # Candidate list: [normalized_query] + reformulations
    candidate_list = [normalized_query] + raw_reformulations
    
    sim_engine = SemanticSimilarity()
    filtered_queries, filter_trace = trace_semantic_filtering(candidate_list, sim_engine)

    # 2. Retrievers
    vec_retriever = VectorRetriever()
    kw_retriever = BM25Retriever()
    hyb_retriever = HybridRetriever(vector_retriever=vec_retriever, keyword_retriever=kw_retriever)

    # 3. Track Ground Truth
    gt_set = target_item.relevant_chunk_set
    gt_tracking: Dict[str, Dict[str, Any]] = {
        cid: {
            "chunk_id": cid,
            "vector_ranks": {},
            "bm25_ranks": {},
            "hybrid_ranks": {},
            "final_rrf_rank": None,
            "final_rrf_score": 0.0,
        }
        for cid in target_item.relevant_chunks
    }

    # 4. Multi-query retrieval tracing
    per_query_traces: List[Dict[str, Any]] = []
    multi_query_fused: Dict[str, Dict[str, Any]] = {}

    for q_idx, q_text in enumerate(filtered_queries):
        # A. Vector
        vec_chunks = vec_retriever.retrieve(q_text, match_count=top_k)
        # B. Keyword
        kw_chunks = kw_retriever.retrieve(q_text, match_count=top_k)
        # C. Hybrid
        hyb_chunks = hyb_retriever.retrieve(q_text, vector_match_count=top_k, keyword_match_count=top_k, k=k_rrf, final_k=top_k)

        # Track GT
        for rank, c in enumerate(vec_chunks, start=1):
            if c.chunk_id in gt_tracking:
                gt_tracking[c.chunk_id]["vector_ranks"][q_text] = rank
        for rank, c in enumerate(kw_chunks, start=1):
            if c.chunk_id in gt_tracking:
                gt_tracking[c.chunk_id]["bm25_ranks"][q_text] = rank
        for rank, c in enumerate(hyb_chunks, start=1):
            if c.chunk_id in gt_tracking:
                gt_tracking[c.chunk_id]["hybrid_ranks"][q_text] = rank

        # Multi-query RRF aggregation
        for rank, c in enumerate(hyb_chunks, start=1):
            contribution = 1.0 / (k_rrf + rank)
            entry = multi_query_fused.setdefault(c.chunk_id, {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "title": c.title,
                "section": c.section,
                "text_snippet": c.chunk_text[:120].replace("\n", " "),
                "is_ground_truth": c.chunk_id in gt_set,
                "score": 0.0,
                "contributions": [],
                "query_appearances": 0,
            })
            entry["score"] += contribution
            entry["contributions"].append((q_text, rank, contribution))
            entry["query_appearances"] += 1

        per_query_traces.append({
            "query": q_text,
            "vector_top_5": [(c.chunk_id, f"{c.score:.4f}", c.title, c.chunk_id in gt_set) for c in vec_chunks[:5]],
            "bm25_top_5": [(c.chunk_id, f"{c.score:.4f}", c.title, c.chunk_id in gt_set) for c in kw_chunks[:5]],
            "hybrid_top_5": [(c.chunk_id, f"{c.score:.4f}", c.title, c.chunk_id in gt_set) for c in hyb_chunks[:5]],
        })

    # 5. Final Ranked List
    final_ranked = sorted(multi_query_fused.values(), key=lambda c: (-c["score"], c["chunk_id"]))
    for final_rank, c in enumerate(final_ranked, start=1):
        if c["chunk_id"] in gt_tracking:
            gt_tracking[c["chunk_id"]]["final_rrf_rank"] = final_rank
            gt_tracking[c["chunk_id"]]["final_rrf_score"] = c["score"]

    diagnosis = {
        "query_id": query_id,
        "raw_query": raw_query,
        "normalized_query": normalized_query,
        "category": target_item.category,
        "relevant_chunks": target_item.relevant_chunks,
        "raw_reformulations": raw_reformulations,
        "filtered_queries": filtered_queries,
        "filter_trace": filter_trace,
        "per_query_traces": per_query_traces,
        "ground_truth_tracking": gt_tracking,
        "final_top_10": final_ranked[:10],
    }

    return diagnosis


def print_diagnostic_report(diag: Dict[str, Any]) -> None:
    """Print a clean diagnostic report to stdout."""
    print("=" * 80)
    print(f"DIAGNOSTIC TRACE FOR QUERY: {diag['query_id']} ({diag['category']})")
    print("=" * 80)
    print(f"Original Query   : {diag['raw_query']}")
    print(f"Normalized Query : {diag['normalized_query']}")
    print(f"Ground Truth     : {diag['relevant_chunks']}")
    print("-" * 80)

    print("1. REFORMULATIONS & FILTERING:")
    for step in diag["filter_trace"]:
        print(f"  [{step['status']:8s}] '{step['candidate']}' -> {step['reason']}")
    print(f"  Active Queries Count: {len(diag['filtered_queries'])}")
    print("-" * 80)

    print("2. GROUND TRUTH RETRIEVAL TRACKING:")
    for cid, info in diag["ground_truth_tracking"].items():
        v_ranks = info["vector_ranks"] or "NOT FOUND in top 20"
        b_ranks = info["bm25_ranks"] or "NOT FOUND in top 20"
        h_ranks = info["hybrid_ranks"] or "NOT FOUND in top 20"
        f_rank = f"Rank {info['final_rrf_rank']}" if info['final_rrf_rank'] else "NOT IN FINAL TOP 20"
        print(f"  Chunk: {cid}")
        print(f"    - Vector Ranks: {v_ranks}")
        print(f"    - BM25 Ranks  : {b_ranks}")
        print(f"    - Hybrid Ranks: {h_ranks}")
        print(f"    - Final RRF   : {f_rank} (Score: {info['final_rrf_score']:.6f})")
    print("-" * 80)

    print("3. FINAL RETRIEVED TOP 10 (Full Pipeline):")
    for rank, c in enumerate(diag["final_top_10"], start=1):
        gt_mark = "[*GROUND TRUTH*]" if c["is_ground_truth"] else ""
        print(f"  #{rank:02d} (score={c['score']:.6f}, queries={c['query_appearances']}) {c['chunk_id']} | {c['document_id']} | {c['title'][:40]} {gt_mark}")
        print(f"      Snippet: {c['text_snippet'][:100]}...")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Reformulation Engine Diagnostic Tracer")
    parser.add_argument("query_id", type=str, help="Query ID to diagnose (e.g. Q004, Q014, Q020)")
    args = parser.parse_args()

    diag = diagnose_query(args.query_id)
    print_diagnostic_report(diag)


if __name__ == "__main__":
    main()
