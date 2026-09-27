"""AI Query Reformulation & Retrieval Presentation Demo.

Terminal-based demonstration for project jury and presentation.
Demonstrates the full end-to-end AI query reformulation and retrieval pipeline:
1. User Query
2. Query Normalization (typo correction, abbreviation expansion, casing)
3. LLM Query Reformulation (Gemini generation)
4. Semantic Filtering (cosine similarity scoring + redundancy filtering + reasons)
5. Final Query Set
6. Baseline Retrieval (original normalized query only)
7. Reformulation Retrieval (multi-query hybrid retrieval fused with RRF)
8. Result Comparison & Rank Movement (ground-truth ranking changes, MRR, NDCG)
9. Cross-Encoder Reranking status (Phase 4 findings)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# Ensure dotenv is loaded first for API keys and Supabase credentials.
# override=True so values in .env win over stale OS-level environment variables.
load_dotenv(override=True)

from embedding.encoder import EmbeddingEncoder
from evaluation.metrics import calculate_all_metrics
from reformulation.cross_encoder import CrossEncoderReranker
from reformulation.llm.client import LLMClient
from reformulation.llm.config import LLMConfig
from reformulation.normalizer import normalize_query
from reformulation.reformulator import QueryReformulator
from reformulation.search_pipeline import SearchPipeline
from reformulation.semantic_similarity import FilterDecision, SemanticSimilarity
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.models import RetrievedChunk
from retrieval.vector_retriever import VectorRetriever

GOLDEN_QUERIES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "evaluation", "golden_queries.json"
)
REFORMULATIONS_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "evaluation", "cache", "reformulations.json"
)
# ms-marco cross-encoder logits: > 0 means the model judges the passage relevant
JUDGE_RELEVANT_THRESHOLD = 0.0

DEMO_MENU_OPTIONS = [
    {
        "id": "1",
        "title": "academic transcript (report card)",
        "query": "academic transcript",
        "golden_id": "Q004",
        "description": "Ranking Bound: target in candidate pool (RRF rank 11) but outside top-10",
    },
    {
        "id": "2",
        "title": "attendence shortage rules",
        "query": "attendence shortage rules",
        "golden_id": "Q016",
        "description": "Typo correction + semantic diversification + RRF ranking",
    },
    {
        "id": "3",
        "title": "hostel allotmnt process",
        "query": "hostel allotmnt process",
        "golden_id": "Q017",
        "description": "Typo correction ('allotmnt') + multi-query hybrid recovery",
    },
    {
        "id": "4",
        "title": "semester withdrawl procedure",
        "query": "semester withdrawl procedure",
        "golden_id": "Q020",
        "description": "Typo correction ('withdrawl') + refund policy retrieval",
    },
    {
        "id": "5",
        "title": "sem fees payment schedule",
        "query": "sem fees payment schedule",
        "golden_id": "Q011",
        "description": "Abbreviation expansion ('sem' -> 'semester') + MRR 0.33 -> 1.00",
    },
]


class PresentationDemo:
    """Orchestrates and displays the live production AI Query Reformulation & Retrieval Demo."""

    def __init__(self, enable_ce: bool = False, use_cache: bool = False):
        print("Initializing production components...")
        self.use_cache = use_cache
        try:
            llm_config = LLMConfig.from_env()
            llm_client = LLMClient(config=llm_config)
            self.reformulator = QueryReformulator(llm_client=llm_client)
            # One embedding model shared by the redundancy filter and vector search.
            encoder = EmbeddingEncoder()
            self.semantic_similarity = SemanticSimilarity(embed_fn=encoder.embed_query)
            self.vector_retriever = VectorRetriever(encoder=encoder)
            self.keyword_retriever = BM25Retriever()
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.vector_retriever,
                keyword_retriever=self.keyword_retriever,
            )
            self.pipeline = SearchPipeline(
                reformulator=self.reformulator,
                semantic_similarity=self.semantic_similarity,
                hybrid_retriever=self.hybrid_retriever,
            )

            # Cross-encoder reranker
            self.reranker = CrossEncoderReranker.from_env()
            if enable_ce:
                self.reranker.enabled = True

        except Exception as e:
            print(f"Failed to initialize components: {e}")
            sys.exit(1)

        # Load golden dataset for ground-truth lookups
        self.golden_queries: List[Dict[str, Any]] = []
        if os.path.exists(GOLDEN_QUERIES_PATH):
            try:
                with open(GOLDEN_QUERIES_PATH, "r", encoding="utf-8") as f:
                    self.golden_queries = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load golden queries: {e}")

        # Index golden queries for fast matching
        self._golden_by_id = {q["query_id"]: q for q in self.golden_queries}
        self._golden_by_text = {q["query"].strip().lower(): q for q in self.golden_queries}

        # Load pre-generated benchmark reformulations cache
        self.reformulation_cache: Dict[str, List[str]] = {}
        if os.path.exists(REFORMULATIONS_CACHE_PATH):
            try:
                with open(REFORMULATIONS_CACHE_PATH, "r", encoding="utf-8") as f:
                    self.reformulation_cache = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load reformulation cache: {e}")

        print("Components ready.\n")

    def find_ground_truth(self, query: str, golden_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Look up ground-truth relevant chunks for a given query."""
        if golden_id and golden_id in self._golden_by_id:
            return self._golden_by_id[golden_id]

        q_clean = query.strip().lower()
        if q_clean in self._golden_by_text:
            return self._golden_by_text[q_clean]

        # Check demo menu options
        for opt in DEMO_MENU_OPTIONS:
            if opt["query"].strip().lower() == q_clean or opt["title"].strip().lower() == q_clean:
                if opt["golden_id"] in self._golden_by_id:
                    return self._golden_by_id[opt["golden_id"]]

        # Try matching by normalized form
        q_norm = normalize_query(query).strip().lower()
        for gq in self.golden_queries:
            if normalize_query(gq["query"]).strip().lower() == q_norm:
                return gq

        return None

    def _judge_relevance(self, query: str, chunks: List[RetrievedChunk]) -> Dict[str, float]:
        """Score chunks against the query with the cross-encoder (label-free relevance judge)."""
        unique = list({c.chunk_id: c for c in chunks}.values())
        try:
            scores = self.reranker.score(query, unique)
        except Exception as e:
            print(f"[!] Relevance judge unavailable ({e}); showing overlap statistics only.")
            print()
            return {}
        return {c.chunk_id: s for c, s in zip(unique, scores)}

    def _print_label_free_comparison(
        self,
        baseline_results: List[RetrievedChunk],
        reform_results: List[RetrievedChunk],
        judge_scores: Dict[str, float],
    ) -> None:
        """[7] for queries without ground truth: judge-based quality + diversity statistics."""
        base_top10 = baseline_results[:10]
        reform_top10 = reform_results[:10]

        print("Ground truth: not labelled for this query — using label-free evaluation.")
        if judge_scores:
            print(f"Relevance judge: Cross-Encoder ({self.reranker.model_name})")
            print(f"  Each result is scored against the query; score > {JUDGE_RELEVANT_THRESHOLD:.1f} counts as relevant.")
        print()

        def judged(chunks: List[RetrievedChunk], k: int) -> List[float]:
            return [judge_scores[c.chunk_id] for c in chunks[:k] if c.chunk_id in judge_scores]

        def precision(chunks: List[RetrievedChunk], k: int) -> float:
            s = judged(chunks, k)
            return sum(x > JUDGE_RELEVANT_THRESHOLD for x in s) / k if s else 0.0

        def mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        rows: List[Tuple[str, str, str]] = []
        if judge_scores:
            rows += [
                ("Judged relevant @5", f"{precision(base_top10, 5) * 5:.0f} / 5", f"{precision(reform_top10, 5) * 5:.0f} / 5"),
                ("Judged relevant @10", f"{precision(base_top10, 10) * 10:.0f} / 10", f"{precision(reform_top10, 10) * 10:.0f} / 10"),
                ("Mean judge score @5", f"{mean(judged(base_top10, 5)):+.3f}", f"{mean(judged(reform_top10, 5)):+.3f}"),
                ("Mean judge score @10", f"{mean(judged(base_top10, 10)):+.3f}", f"{mean(judged(reform_top10, 10)):+.3f}"),
                ("Top-1 judge score", f"{mean(judged(base_top10, 1)):+.3f}", f"{mean(judged(reform_top10, 1)):+.3f}"),
            ]
        rows.append((
            "Distinct documents @10",
            str(len({c.document_id for c in base_top10})),
            str(len({c.document_id for c in reform_top10})),
        ))

        print(f"{'Metric':<25} | {'Original':<14} | {'Reformulated':<14}")
        print("-" * 58)
        for name, b, r in rows:
            print(f"{name:<25} | {b:<14} | {r:<14}")
        print("-" * 58)

        base_ids = {c.chunk_id for c in base_top10}
        reform_ids = {c.chunk_id for c in reform_top10}
        overlap = len(base_ids & reform_ids)
        print(f"Top-10 overlap            : {overlap}/10 chunks shared "
              f"({len(reform_ids - base_ids)} new surfaced by reformulation)")

        if judge_scores:
            delta = mean(judged(reform_top10, 10)) - mean(judged(base_top10, 10))
            if delta > 0.1:
                verdict = f"▲ Reformulation improved judged relevance of the top-10 (+{delta:.3f})"
            elif delta < -0.1:
                verdict = f"▼ Reformulation reduced judged relevance of the top-10 ({delta:.3f})"
            else:
                verdict = f"— Comparable judged relevance ({delta:+.3f})"
            print(f"Verdict                   : {verdict}")

    def _print_label_free_rank_movement(
        self,
        baseline_results: List[RetrievedChunk],
        reform_results: List[RetrievedChunk],
        judge_scores: Dict[str, float],
    ) -> None:
        """[8] for queries without ground truth: how each top chunk moved between configurations."""
        base_rank = {c.chunk_id: r for r, c in enumerate(baseline_results, start=1)}
        reform_rank = {c.chunk_id: r for r, c in enumerate(reform_results, start=1)}

        # Everything in either top-10, ordered by reformulated rank (absent ones last)
        shown = list(dict.fromkeys([c.chunk_id for c in reform_results[:10]] + [c.chunk_id for c in baseline_results[:10]]))
        shown.sort(key=lambda cid: (reform_rank.get(cid, 999), base_rank.get(cid, 999)))

        def fmt(r: Optional[int]) -> str:
            return str(r) if r is not None else ">20"

        promoted = demoted = new = dropped = 0
        print(f"{'Chunk ID':<28} | {'Original':>8} → {'Reform':<6} | {'Movement':<14} | {'Judge':>7}")
        print("-" * 78)
        for cid in shown:
            rb, rr = base_rank.get(cid), reform_rank.get(cid)
            if rb is None:
                move, new = "★ NEW", new + 1
            elif rr is None:
                move, dropped = "✗ DROPPED", dropped + 1
            elif rb > rr:
                move, promoted = f"▲ +{rb - rr}", promoted + 1
            elif rb < rr:
                move, demoted = f"▼ -{rr - rb}", demoted + 1
            else:
                move = "— same"
            js = judge_scores.get(cid)
            js_str = f"{js:+.2f}" if js is not None else "n/a"
            if js is not None and js > JUDGE_RELEVANT_THRESHOLD:
                js_str += " ✓"
            print(f"{cid:<28} | {fmt(rb):>8} → {fmt(rr):<6} | {move:<14} | {js_str:>7}")
        print("-" * 78)
        print(f"Summary: {promoted} promoted, {demoted} demoted, {new} newly surfaced, {dropped} dropped out of top-20")
        if judge_scores:
            new_rel = sum(
                1 for cid in shown
                if cid not in base_rank and judge_scores.get(cid, -99) > JUDGE_RELEVANT_THRESHOLD
            )
            lost_rel = sum(
                1 for cid in shown
                if cid not in reform_rank and judge_scores.get(cid, -99) > JUDGE_RELEVANT_THRESHOLD
            )
            print(f"         {new_rel} judged-relevant chunk(s) gained, {lost_rel} judged-relevant chunk(s) lost")
        print()

    def execute_demo_query(self, query: str, golden_id: Optional[str] = None) -> None:
        """Run the full presentation flow for a single query."""
        gt = self.find_ground_truth(query, golden_id)
        if gt and not golden_id:
            golden_id = gt.get("query_id")

        print()
        print("============================================================")
        print("        AI QUERY REFORMULATION & SEARCH DEMO")
        print("============================================================")
        print()
        print("USER QUERY")
        print(f"> {query}")
        print()

        # ------------------------------------------------------------
        # [1] NORMALIZATION
        # ------------------------------------------------------------
        print("-" * 60)
        print("[1] NORMALIZATION")
        print("-" * 60)
        print()
        normalized = normalize_query(query)
        print("Original:")
        print(f"  {query}")
        print()
        print("Normalized:")
        print(f"  {normalized}")
        print()

        is_changed = query.strip() != normalized.strip()
        if is_changed:
            changes = []
            if query.strip().lower() != normalized.strip().lower():
                changes.append("spelling correction")
            if query.strip() != normalized.strip():
                changes.append("casing / whitespace / punctuation normalization")
            print(f"Status: CHANGED ({' & '.join(changes)})")
        else:
            print("Status: UNCHANGED (query was already in canonical normalized form)")
        print()

        # ------------------------------------------------------------
        # [2] LLM QUERY REFORMULATION
        # ------------------------------------------------------------
        print("-" * 60)
        print("[2] LLM QUERY REFORMULATION")
        print("-" * 60)
        print()
        print("Generated Reformulations:")
        print()

        reformulations: List[str] = []
        if self.use_cache and golden_id and golden_id in self.reformulation_cache:
            reformulations = self.reformulation_cache[golden_id][:5]
        else:
            try:
                reformulations = self.reformulator.generate_variants(query, max_variants=5)
            except Exception as e:
                # Graceful fallback to pre-generated benchmark cache if live LLM rate-limits or fails
                if golden_id and golden_id in self.reformulation_cache:
                    print(f"  [!] Live LLM call encountered an issue ({e}).")
                    print(f"  [i] Using pre-generated benchmark reformulations for {golden_id}:")
                    reformulations = self.reformulation_cache[golden_id][:5]
                else:
                    q_clean = query.strip().lower()
                    matched_qid = next(
                        (qid for qid, gq in self._golden_by_id.items() if gq["query"].strip().lower() == q_clean),
                        None,
                    )
                    if matched_qid and matched_qid in self.reformulation_cache:
                        print(f"  [!] Live LLM call encountered an issue ({e}).")
                        print(f"  [i] Using pre-generated benchmark reformulations for {matched_qid}:")
                        reformulations = self.reformulation_cache[matched_qid][:5]
                    else:
                        print(f"  [!] LLM error (using original query only): {e}")
                        reformulations = []

        if reformulations:
            for i, v in enumerate(reformulations, start=1):
                print(f"  {i}. {v}")
        else:
            print("  (No variants generated; pipeline falls back safely to original query)")
        print()

        # ------------------------------------------------------------
        # [3] SEMANTIC FILTERING
        # ------------------------------------------------------------
        print("-" * 60)
        print("[3] SEMANTIC FILTERING")
        print("-" * 60)
        print()
        print("Candidate Evaluation (Semantic Cosine Similarity & Lexical Diversity):")
        print()
        candidate_queries = [normalized] + reformulations

        try:
            final_queries, filter_decisions = (
                self.semantic_similarity.filter_redundant_with_diagnostics(
                    candidate_queries, original_query=normalized
                )
            )
        except Exception as e:
            print(f"[!] Semantic filter error: {e}")
            final_queries = candidate_queries
            filter_decisions = [
                FilterDecision(query=q, similarity=1.0, kept=True, reason="Fallback")
                for q in candidate_queries
            ]

        for idx, dec in enumerate(filter_decisions, start=1):
            status_str = "✓ KEPT" if dec.kept else "✗ REMOVED"
            print(f"  [{idx}] {dec.query}")
            print(f"      Similarity : {dec.similarity:.4f}")
            print(f"      Status     : {status_str}")
            print(f"      Reason     : {dec.reason}")
            if dec.compared_with:
                print(f"      Compared with: {dec.compared_with}")
            print()

        # ------------------------------------------------------------
        # [4] FINAL QUERY SET
        # ------------------------------------------------------------
        print("-" * 60)
        print("[4] FINAL QUERY SET")
        print("-" * 60)
        print()
        print("FINAL QUERIES SENT TO RETRIEVAL:")
        for idx, q_item in enumerate(final_queries, start=1):
            print(f"  {idx}. {q_item}")
        print()
        retained_count = len(final_queries)
        removed_count = len(filter_decisions) - retained_count
        print(f"  Original query           : 1")
        print(f"  Reformulations generated : {len(reformulations)}")
        print(f"  Queries retained         : {retained_count}")
        print(f"  Queries removed          : {removed_count}")
        print()

        # Check ground truth
        gt = self.find_ground_truth(query, golden_id)
        relevant_set: Set[str] = set(gt["relevant_chunks"]) if gt else set()

        # ------------------------------------------------------------
        # [5] BASELINE RETRIEVAL (ORIGINAL QUERY ONLY)
        # ------------------------------------------------------------
        print("-" * 60)
        print("[5] BASELINE: ORIGINAL QUERY ONLY")
        print("-" * 60)
        print()
        print("Query:")
        print(f"  {normalized}")
        print()
        print("Retrieval Strategy: Single-Query Hybrid (Vector + Keyword) fused via RRF (k=60)")
        print()
        print("Top retrieved results:")
        print()

        try:
            baseline_results = self.hybrid_retriever.retrieve(
                query=normalized,
                vector_match_count=20,
                keyword_match_count=20,
                k=60,
                final_k=20,
            )
        except Exception as e:
            print(f"[!] Baseline retrieval failed: {e}")
            baseline_results = []

        for rank, chunk in enumerate(baseline_results[:5], start=1):
            is_gt = chunk.chunk_id in relevant_set
            gt_tag = " [✓ GROUND TRUTH TARGET]" if is_gt else ""
            print(f"  Rank {rank} | Score: {chunk.score:.5f}{gt_tag}")
            print(f"  Document : {chunk.document_id} | Chunk: {chunk.chunk_id}")
            print(f"  Title    : {chunk.title}")
            if chunk.section:
                print(f"  Section  : {chunk.section}")
            snippet = (chunk.chunk_text or "").strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            print(f"  Snippet  : {snippet}")
            print()

        if gt:
            base_hit_ranks = [
                r for r, c in enumerate(baseline_results, start=1) if c.chunk_id in relevant_set
            ]
            if base_hit_ranks:
                print(f"Ground Truth Status: Relevant target found at Rank {base_hit_ranks[0]}")
            else:
                print("Ground Truth Status: No relevant target found in top-20 results")
            print()

        # ------------------------------------------------------------
        # [6] REFORMULATION RETRIEVAL
        # ------------------------------------------------------------
        print("-" * 60)
        print("[6] REFORMULATION + HYBRID RETRIEVAL")
        print("-" * 60)
        print()
        print("Queries used:")
        for idx, q_item in enumerate(final_queries, start=1):
            print(f"  {idx}. {q_item}")
        print()
        print("Retrieval Strategy: Multi-Query Hybrid (Vector + Keyword) fused with Reciprocal Rank Fusion (RRF, k=60)")
        print()
        print("Top retrieved results:")
        print()

        try:
            reform_results = self.pipeline.aggregate_retrieval(
                queries=final_queries,
                vector_match_count=20,
                keyword_match_count=20,
                k=60,
                final_k=20,
            )
        except Exception as e:
            print(f"[!] Reformulation retrieval failed: {e}")
            reform_results = []

        for rank, chunk in enumerate(reform_results[:5], start=1):
            is_gt = chunk.chunk_id in relevant_set
            gt_tag = " [✓ GROUND TRUTH TARGET]" if is_gt else ""
            print(f"  Rank {rank} | Score: {chunk.score:.5f}{gt_tag}")
            print(f"  Document : {chunk.document_id} | Chunk: {chunk.chunk_id}")
            print(f"  Title    : {chunk.title}")
            if chunk.section:
                print(f"  Section  : {chunk.section}")
            snippet = (chunk.chunk_text or "").strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            print(f"  Snippet  : {snippet}")
            print()

        if gt:
            reform_hit_ranks = [
                r for r, c in enumerate(reform_results, start=1) if c.chunk_id in relevant_set
            ]
            if reform_hit_ranks:
                print(f"Ground Truth Status: Relevant target found at Rank {reform_hit_ranks[0]}")
            else:
                print("Ground Truth Status: No relevant target found in top-20 results")
            print()

        # ------------------------------------------------------------
        # [7] RESULT COMPARISON
        # ------------------------------------------------------------
        print("-" * 60)
        print("[7] RESULT COMPARISON")
        print("-" * 60)
        print()
        judge_scores: Dict[str, float] = {}
        if not gt:
            judge_scores = self._judge_relevance(normalized, baseline_results[:10] + reform_results[:10])
            self._print_label_free_comparison(baseline_results, reform_results, judge_scores)
        else:
            base_top10 = baseline_results[:10]
            reform_top10 = reform_results[:10]
            base_ids = [c.chunk_id for c in base_top10]
            reform_ids = [c.chunk_id for c in reform_top10]

            m_base = calculate_all_metrics(
                base_ids, gt["relevant_chunks"], gt.get("graded_relevance"), k_list=[5, 10]
            )
            m_reform = calculate_all_metrics(
                reform_ids, gt["relevant_chunks"], gt.get("graded_relevance"), k_list=[5, 10]
            )

            base_hit_ranks = [
                r for r, c in enumerate(baseline_results, start=1) if c.chunk_id in relevant_set
            ]
            reform_hit_ranks = [
                r for r, c in enumerate(reform_results, start=1) if c.chunk_id in relevant_set
            ]

            base_rank_str = str(base_hit_ranks[0]) if base_hit_ranks else "Not in top 20"
            reform_rank_str = str(reform_hit_ranks[0]) if reform_hit_ranks else "Not in top 20"

            f5_base = "YES" if m_base["recall@5"] > 0 else "NO"
            f5_ref = "YES" if m_reform["recall@5"] > 0 else "NO"
            f10_base = "YES" if m_base["recall@10"] > 0 else "NO"
            f10_ref = "YES" if m_reform["recall@10"] > 0 else "NO"

            print(f"{'Metric':<25} | {'Original':<14} | {'Reformulated':<14}")
            print("-" * 58)
            print(f"{'Relevant found @5':<25} | {f5_base:<14} | {f5_ref:<14}")
            print(f"{'Relevant found @10':<25} | {f10_base:<14} | {f10_ref:<14}")
            print(f"{'First relevant rank':<25} | {base_rank_str:<14} | {reform_rank_str:<14}")
            print(f"{'MRR@10':<25} | {m_base['mrr']:<14.4f} | {m_reform['mrr']:<14.4f}")
            print(f"{'NDCG@5':<25} | {m_base['ndcg@5']:<14.4f} | {m_reform['ndcg@5']:<14.4f}")
            print(f"{'NDCG@10':<25} | {m_base['ndcg@10']:<14.4f} | {m_reform['ndcg@10']:<14.4f}")
            print(f"{'Recall@10':<25} | {m_base['recall@10']:<14.4f} | {m_reform['recall@10']:<14.4f}")
            print("-" * 58)
        print()

        # ------------------------------------------------------------
        # [8] RANK MOVEMENT
        # ------------------------------------------------------------
        print("-" * 60)
        print("[8] RANK MOVEMENT")
        print("-" * 60)
        print()
        if not gt:
            self._print_label_free_rank_movement(baseline_results, reform_results, judge_scores)
        else:
            for target_chunk in gt["relevant_chunks"]:
                r_base = next(
                    (r for r, c in enumerate(baseline_results, start=1) if c.chunk_id == target_chunk),
                    None,
                )
                r_reform = next(
                    (r for r, c in enumerate(reform_results, start=1) if c.chunk_id == target_chunk),
                    None,
                )

                print(f"Target Relevant Chunk: {target_chunk}")
                base_str = str(r_base) if r_base is not None else "Absent (>20)"
                reform_str = str(r_reform) if r_reform is not None else "Absent (>20)"
                print(f"  Original query rank     : {base_str}")
                print(f"  Reformulated query rank : {reform_str}")

                if r_base is not None and r_reform is not None:
                    delta = r_base - r_reform
                    if delta > 0:
                        print(f"  Rank movement           : {r_base} → {r_reform}")
                        print(f"  Improvement             : ▲ +{delta} positions")
                    elif delta < 0:
                        print(f"  Rank movement           : {r_base} → {r_reform}")
                        print(f"  Regression              : ▼ {delta} positions")
                    else:
                        print(f"  Rank movement           : {r_base} → {r_reform}")
                        print(f"  Movement delta          : — Unchanged")
                elif r_base is None and r_reform is not None:
                    print(f"  Rank movement           : Absent (>20) → {r_reform}")
                    print(f"  Improvement             : ▲ Recovered into top results at Rank {r_reform}")
                elif r_base is not None and r_reform is None:
                    print(f"  Rank movement           : {r_base} → Absent (>20)")
                    print(f"  Regression              : ▼ Dropped outside top results")
                else:
                    print(f"  Rank movement           : Absent (>20) → Absent (>20)")
                    print(f"  Status                  : Not retrieved in top-20 by either configuration")
                print()

        # ------------------------------------------------------------
        # [9] CROSS-ENCODER RERANKING
        # ------------------------------------------------------------
        print("-" * 60)
        print("[9] CROSS-ENCODER RERANKING")
        print("-" * 60)
        print()
        if self.reranker.enabled:
            print("Status: ENABLED (live second-stage reranking active)")
            print(f"Model          : {self.reranker.model_name}")
            print("Candidate pool : K = 30")
            print()
            print("Stage Distinction:")
            print("  Stage 1: Multi-Query Hybrid RRF Candidate Generation")
            print("  Stage 2: Cross-Encoder Deep Relevance Reranking")
            print()
            try:
                candidate_pool = self.pipeline.aggregate_retrieval(
                    queries=final_queries,
                    vector_match_count=20,
                    keyword_match_count=20,
                    k=60,
                    final_k=30,
                )
                ce_ranked = self.reranker.rerank(query, candidate_pool)

                print(
                    f"{'Final Rank':<11} | {'RRF Rank':<9} | {'Chunk ID':<25} | {'CE Score':<10} | {'Ground Truth':<12}"
                )
                print("-" * 75)
                for ce_rank, chunk in enumerate(ce_ranked[:10], start=1):
                    rrf_rank = next(
                        (idx for idx, c in enumerate(candidate_pool, start=1) if c.chunk_id == chunk.chunk_id),
                        "N/A",
                    )
                    ce_score = chunk.metadata.get("ce_score")
                    ce_str = f"{ce_score:+.4f}" if ce_score is not None else "N/A"
                    is_gt = "✓ YES" if chunk.chunk_id in relevant_set else "no"
                    print(f"{ce_rank:<11} | {str(rrf_rank):<9} | {chunk.chunk_id:<25} | {ce_str:<10} | {is_gt:<12}")
                print()
            except Exception as e:
                print(f"[!] Cross-Encoder reranking failed: {e}")
        else:
            print("Status: DISABLED in production pipeline configuration (RERANKER_ENABLED=false)")
            print()
            print("Benchmark Findings (50 golden queries):")
            print("  • Phase 4: pure Cross-Encoder reranking lowered Recall@10 (0.69 → 0.62) and MRR (0.56 → 0.53);")
            print("    14 queries improved, 17 regressed. Cost: ~4.0s CPU per query for 50 candidates.")
            print("  • Phase 5: contextual embeddings + BM25 raised candidate Recall@50 from 87% to 96%")
            print("    and the full pipeline to Recall@10 0.83, MRR 0.666 (from 0.69 / 0.561).")
            print("  • Conclusion: kept as an optional stage (RERANKER_ENABLED=true); to be re-evaluated")
            print("    on the stronger Phase 5 candidate pool.")
        print()

    def run(self) -> None:
        """Run interactive presentation loop."""
        while True:
            print("=" * 60)
            print("SELECT DEMO QUERY")
            print("=" * 60)
            print()
            for opt in DEMO_MENU_OPTIONS:
                print(f"{opt['id']}. {opt['title']:<35} ({opt['description']})")
            print(f"{len(DEMO_MENU_OPTIONS) + 1}. Enter custom query")
            print(f"{len(DEMO_MENU_OPTIONS) + 2}. Exit")
            print()

            try:
                choice = input("Choice: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting demo.")
                break

            if not choice:
                continue

            if choice == str(len(DEMO_MENU_OPTIONS) + 2) or choice.lower() in ("exit", "quit"):
                print("Exiting demo.")
                break

            if choice == str(len(DEMO_MENU_OPTIONS) + 1):
                try:
                    custom_q = input("\nEnter custom query: ").strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if not custom_q:
                    continue
                self.execute_demo_query(custom_q)
            else:
                selected_opt = next((opt for opt in DEMO_MENU_OPTIONS if opt["id"] == choice), None)
                if selected_opt:
                    self.execute_demo_query(selected_opt["query"], golden_id=selected_opt["golden_id"])
                else:
                    # Allow user to type the query directly
                    self.execute_demo_query(choice)

            print()
            input("Press Enter to continue...")
            print("\n" * 2)


def main():
    parser = argparse.ArgumentParser(description="AI Query Reformulation & Retrieval Presentation Demo")
    parser.add_argument("--query", "-q", type=str, default=None, help="Directly run a single query demo")
    parser.add_argument("--enable-ce", "--reranker", action="store_true", help="Enable live Cross-Encoder reranking")
    parser.add_argument("--use-cache", action="store_true", help="Use pre-generated benchmark reformulations cache")
    args = parser.parse_args()

    demo = PresentationDemo(enable_ce=args.enable_ce, use_cache=args.use_cache)
    if args.query:
        demo.execute_demo_query(args.query)
    else:
        demo.run()


if __name__ == "__main__":
    main()
