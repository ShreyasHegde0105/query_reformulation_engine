"""AI Query Reformulation & Retrieval Presentation Demo.

Terminal-based demonstration for project jury and presentation.
Demonstrates the full end-to-end AI query reformulation and retrieval pipeline:
1. User Query
2. Query Normalization (typo correction, abbreviation expansion, casing)
3. LLM Query Reformulation (multi-provider, key fail-over)
4. Semantic Filtering (cosine similarity scoring + redundancy filtering + reasons)
5. Final Query Set
6. Baseline Retrieval (original normalized query only)
7. Reformulation Retrieval (multi-query hybrid retrieval fused with RRF)
8. Result Comparison & Rank Movement (labelled metrics only; nothing is judged without labels)
9. Production reranking (optional, --enable-ce): contextual cross-encoder, logits shown as
   uncalibrated ordering scores
10. Answerability: what the system can and cannot tell
11. Optional LLM judge diagnostics (--judge): evaluation-only, never ground truth, never a verdict
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
from reformulation.production_reranker import LOGIT_KEY, ProductionReranker, RerankerConfig
from reformulation.llm.fallback import FallbackLLMClient
from reformulation.normalizer import normalize_query
from reformulation.spelling import CorpusSpellCorrector
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
# Candidate pool per query for multi-query fusion, as in the production API.
CANDIDATE_K = 50

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

    def __init__(self, enable_ce: bool = False, use_cache: bool = False, enable_judge: bool = False):
        print("Initializing production components...")
        self.use_cache = use_cache
        self.enable_judge = enable_judge
        try:
            llm_client = FallbackLLMClient.from_env()
            print(f"LLM keys (fail-over order): {', '.join(llm_client.labels)}")
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
            # Same corpus spell corrector as the production API (api/app.py).
            self.spell_corrector = CorpusSpellCorrector.from_chunks_file()
            self.pipeline = SearchPipeline(
                reformulator=self.reformulator,
                semantic_similarity=self.semantic_similarity,
                hybrid_retriever=self.hybrid_retriever,
                spell_corrector=self.spell_corrector,
            )

            # Production reranker (contextual cross-encoder); off unless --enable-ce or RERANKER_ENABLED=true.
            cfg = RerankerConfig.from_env()
            self.reranker = ProductionReranker(dataclasses.replace(cfg, enabled=cfg.enabled or enable_ce))

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

        # Live LLM rewrites from this session, so repeating a query does not use quota again.
        self._live_reformulations: Dict[str, List[str]] = {}

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

    @staticmethod
    def _labelled_metrics(ranked: List[RetrievedChunk], gt: Dict[str, Any]) -> Dict[str, float]:
        ids = [c.chunk_id for c in ranked]
        rel = set(gt["relevant_chunks"])
        m = calculate_all_metrics(ids, gt["relevant_chunks"], gt.get("graded_relevance"), k_list=[5, 10])
        m["p@5"] = len(rel & set(ids[:5])) / 5
        m["p@10"] = len(rel & set(ids[:10])) / 10
        m["mrr@10"] = next((1 / i for i, c in enumerate(ids[:10], 1) if c in rel), 0.0)
        return m

    def _print_labelled_comparison(self, systems: Dict[str, List[RetrievedChunk]], gt: Dict[str, Any]) -> None:
        """[7] Metrics from the query's labels only (labels: evaluation/golden_queries.json)."""
        print(f"Labels: {gt['query_id']} in evaluation/golden_queries.json "
              "(graded 0-3; label provenance is not recorded in the repository).")
        if gt.get("description"):
            print(f"Label note (evidence for the grades): {gt['description']}")
        print()
        names = list(systems)
        metrics = {n: self._labelled_metrics(r, gt) for n, r in systems.items()}
        print(f"{'Metric':<14} | " + " | ".join(f"{n:<13}" for n in names))
        print("-" * (17 + 16 * len(names)))
        for key, label in (("p@5", "Precision@5"), ("p@10", "Precision@10"), ("recall@5", "Recall@5"),
                           ("recall@10", "Recall@10"), ("mrr@10", "MRR@10"), ("ndcg@10", "NDCG@10")):
            print(f"{label:<14} | " + " | ".join(f"{metrics[n][key]:<13.4f}" for n in names))
        print()
        print("One query cannot show that a system is better; see evaluation/results/ad_benchmark_*.md "
              "for measured results over full query sets.")
        graded = gt.get("graded_relevance") or {c: 2 for c in gt["relevant_chunks"]}
        print()
        print("Labelled grade of each top-5 result (- = not labelled):")
        for n in names:
            grades = [str(graded.get(c.chunk_id, "-")) for c in systems[n][:5]]
            print(f"  {n:<13}: " + "  ".join(grades))

    def _print_unlabelled_comparison(self, baseline: List[RetrievedChunk], reform: List[RetrievedChunk]) -> None:
        """[7] for queries without labels: overlap and diversity only. Relevance is NOT judged."""
        print("Relevance: NOT EVALUATED. This query has no labels, and no model score is used as a")
        print("substitute (cross-encoder logits are uncalibrated and are not relevance judgments).")
        print()
        b10, r10 = baseline[:10], reform[:10]
        b_ids, r_ids = {c.chunk_id for c in b10}, {c.chunk_id for c in r10}
        print(f"{'Statistic':<28} | {'Original':<10} | {'Reformulated':<12}")
        print("-" * 58)
        print(f"{'Distinct documents in top 10':<28} | {len({c.document_id for c in b10}):<10} | "
              f"{len({c.document_id for c in r10}):<12}")
        print("-" * 58)
        print(f"Top-10 overlap: {len(b_ids & r_ids)}/10 shared; {len(r_ids - b_ids)} in the reformulated "
              f"top 10 are not in the original top 10.")

    @staticmethod
    def _print_unlabelled_rank_movement(baseline: List[RetrievedChunk], reform: List[RetrievedChunk]) -> None:
        """[8] for queries without labels: rank changes only (top-20 lists)."""
        base_rank = {c.chunk_id: r for r, c in enumerate(baseline, start=1)}
        reform_rank = {c.chunk_id: r for r, c in enumerate(reform, start=1)}
        shown = list(dict.fromkeys([c.chunk_id for c in reform[:10]] + [c.chunk_id for c in baseline[:10]]))
        shown.sort(key=lambda cid: (reform_rank.get(cid, 999), base_rank.get(cid, 999)))
        counts = {"promoted": 0, "demoted": 0, "new": 0, "dropped": 0}
        print(f"{'Chunk ID':<28} | {'Original':>8} -> {'Reform':<6} | Movement")
        print("-" * 64)
        for cid in shown:
            rb, rr = base_rank.get(cid), reform_rank.get(cid)
            if rb is None:
                move = "NEW (not in original top 20)"
                counts["new"] += 1
            elif rr is None:
                move = "DROPPED (not in reformulated top 20)"
                counts["dropped"] += 1
            elif rb > rr:
                move = f"up {rb - rr}"
                counts["promoted"] += 1
            elif rb < rr:
                move = f"down {rr - rb}"
                counts["demoted"] += 1
            else:
                move = "same"
            print(f"{cid:<28} | {str(rb) if rb else '>20':>8} -> {str(rr) if rr else '>20':<6} | {move}")
        print("-" * 64)
        print(f"Summary (top-20 lists): {counts['promoted']} promoted, {counts['demoted']} demoted, "
              f"{counts['new']} new, {counts['dropped']} dropped.")
        print("Rank movement is not evidence of better or worse relevance.")
        print()

    def _print_llm_judge(self, query: str, query_id: str, pools: Dict[str, List[RetrievedChunk]]) -> None:
        """[11] Optional evaluation-only LLM judge. Diagnostic; never ground truth, never a verdict."""
        try:
            from evaluation.judge import JudgeConfig, LLMRelevanceJudge
            cfg = JudgeConfig.from_env()
            judge = LLMRelevanceJudge(cfg.build_client(), cfg)
        except Exception as e:
            print(f"LLM judge unavailable ({e}).")
            return
        pool = list({c.chunk_id: c for lst in pools.values() for c in lst[:5]}.values())
        res = judge.judge(query_id, query, pool)
        print(f"Judge: {judge.provider}/{judge.model} | rubric {res.rubric_version} | prompt {res.prompt_version} | "
              "blind to which list each passage came from.")
        if res.status != "ok":
            print(f"Judgment rejected ({res.error}); no grades shown.")
            return
        grades = {j.chunk_id: j for j in res.judgments}
        print(f"Intents: {res.query_intents} | answerability among these passages: {res.answerability}")
        for name, lst in pools.items():
            print(f"  {name:<13}: " + "  ".join(str(grades[c.chunk_id].grade) if c.chunk_id in grades else "-"
                                              for c in lst[:5]))
        for j in res.judgments:
            if j.grade > 0:
                print(f"  {j.chunk_id} grade {j.grade}: evidence '{j.evidence[0][:100]}'")
        print("These grades are an LLM opinion for diagnosis only. They are not labels and do not show")
        print("that either list is better.")

    def execute_demo_query(self, query: str, golden_id: Optional[str] = None) -> List[RetrievedChunk]:
        """Run the full presentation flow for a single query.

        Prints every step and returns the reformulated pipeline's ranked results
        (used by ``demo_server.py`` to show them in the browser).
        """
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
        rules_only = normalize_query(query)
        # The query used for retrieval, as in the API: rules + corpus spelling correction.
        normalized = normalize_query(query, spell_corrector=self.spell_corrector)
        print("Original:")
        print(f"  {query}")
        print()
        print("Normalized (rules):")
        print(f"  {rules_only}")
        if normalized != rules_only:
            print("Corrected (corpus spelling, used for retrieval):")
            print(f"  {normalized}")
        else:
            print("Corrected: no corpus spelling correction applied (the corrector only fixes words one edit")
            print("  away from a corpus word; longer typos are left to the LLM rewrites).")
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
        live_key = normalized.strip().lower()
        if self.use_cache and golden_id and golden_id in self.reformulation_cache:
            reformulations = self.reformulation_cache[golden_id][:5]
            print(f"  [i] Saved benchmark reformulations for {golden_id} (no LLM call):")
        elif live_key in self._live_reformulations:
            reformulations = self._live_reformulations[live_key]
            print("  [i] Reusing this session's LLM reformulations (no new LLM call):")
        else:
            try:
                reformulations = self.reformulator.generate_variants(query, max_variants=5)
                self._live_reformulations[live_key] = reformulations
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

        t_base = time.perf_counter()
        try:
            # Candidate pool of CANDIDATE_K (as in the API); the top 20 are shown and compared.
            baseline_pool = self.hybrid_retriever.retrieve(
                query=normalized,
                vector_match_count=20,
                keyword_match_count=20,
                k=60,
                final_k=CANDIDATE_K,
            )
        except Exception as e:
            print(f"[!] Baseline retrieval failed: {e}")
            baseline_pool = []
        baseline_ms = (time.perf_counter() - t_base) * 1000
        baseline_results = baseline_pool[:20]
        print(f"Retrieval latency: {baseline_ms:.0f} ms")
        print()

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

        t_ref = time.perf_counter()
        try:
            reform_pool = self.pipeline.aggregate_retrieval(
                queries=final_queries,
                vector_match_count=20,
                keyword_match_count=20,
                k=60,
                final_k=CANDIDATE_K,
                candidate_k=CANDIDATE_K,
            )
        except Exception as e:
            print(f"[!] Reformulation retrieval failed: {e}")
            reform_pool = []
        reform_ms = (time.perf_counter() - t_ref) * 1000
        reform_results = reform_pool[:20]
        print(f"Retrieval latency: {reform_ms:.0f} ms (excludes the LLM rewrite call above)")
        print()

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
        # Production reranking (computed first so section [7] can include it).
        reranked_results: List[RetrievedChunk] = []
        rerank_outcome = None
        if self.reranker.enabled:
            rerank_outcome = self.reranker.rerank(normalized, reform_pool, final_k=20)
            reranked_results = rerank_outcome.results

        print("-" * 60)
        print("[7] RESULT COMPARISON")
        print("-" * 60)
        print()
        systems = {"Original": baseline_results, "Reformulated": reform_results}
        if reranked_results:
            systems["Reranked"] = reranked_results
        if gt:
            self._print_labelled_comparison(systems, gt)
        else:
            self._print_unlabelled_comparison(baseline_results, reform_results)
        print()
        print(f"Latency: original retrieval {baseline_ms:.0f} ms | reformulated retrieval {reform_ms:.0f} ms"
              + (f" | reranking {rerank_outcome.latency_ms:.0f} ms" if rerank_outcome else ""))
        print()

        # ------------------------------------------------------------
        # [8] RANK MOVEMENT
        # ------------------------------------------------------------
        print("-" * 60)
        print("[8] RANK MOVEMENT")
        print("-" * 60)
        print()
        if not gt:
            self._print_unlabelled_rank_movement(baseline_results, reform_results)
        else:
            for target_chunk in gt["relevant_chunks"]:
                ranks = {name: next((r for r, c in enumerate(lst, start=1) if c.chunk_id == target_chunk), None)
                         for name, lst in systems.items()}
                print(f"Labelled relevant chunk: {target_chunk}")
                for name, r in ranks.items():
                    print(f"  {name:<13} rank: {r if r is not None else 'not in top 20'}")
                print()

        # ------------------------------------------------------------
        # [9] PRODUCTION RERANKING
        # ------------------------------------------------------------
        print("-" * 60)
        print("[9] PRODUCTION RERANKING")
        print("-" * 60)
        print()
        if rerank_outcome is not None:
            print(f"Model: {self.reranker.config.model_name} | candidates scored: {rerank_outcome.candidates_scored} "
                  f"| latency: {rerank_outcome.latency_ms:.0f} ms")
            print("Scores below are raw cross-encoder logits: they order results for THIS query only.")
            print("They are not probabilities and no relevance threshold is applied.")
            if rerank_outcome.error:
                print(f"[!] Reranking failed ({rerank_outcome.error}); retrieval order kept.")
            print()
            pool_rank = {c.chunk_id: r for r, c in enumerate(reform_pool, start=1)}
            graded = (gt.get("graded_relevance") or {}) if gt else {}
            print(f"{'Rank':<5} | {'Fusion rank':<11} | {'Chunk ID':<28} | {'Logit':>8} | Labelled grade")
            print("-" * 75)
            for r, c in enumerate(reranked_results[:10], start=1):
                logit = (c.metadata or {}).get(LOGIT_KEY)
                lab = str(graded.get(c.chunk_id, "-")) if gt else "no labels"
                print(f"{r:<5} | {pool_rank.get(c.chunk_id, '?'):<11} | {c.chunk_id:<28} | "
                      f"{(f'{logit:+.2f}' if logit is not None else 'n/a'):>8} | {lab}")
        else:
            print("Status: DISABLED (RERANKER_ENABLED=false; enable for this demo with --enable-ce).")
            print("Measured results: evaluation/results/ad_benchmark_*.md (configurations A-D). The reranker")
            print("stays off in production until a benchmark shows a repeatable benefit.")
        print()

        # ------------------------------------------------------------
        # [10] ANSWERABILITY
        # ------------------------------------------------------------
        print("-" * 60)
        print("[10] ANSWERABILITY")
        print("-" * 60)
        print()
        if gt and not gt.get("relevant_chunks"):
            print("Labels mark this query as unanswerable from the corpus.")
        print("Unanswerable-query detection: NOT AVAILABLE. The system has no calibrated way to tell that the")
        print("corpus does not answer a query; it always returns its best-matching chunks. Reranker logits")
        print("are uncalibrated and are not used as a no-answer signal. Treat results for questions about")
        print("topics the corpus may not cover with care.")
        print()

        # ------------------------------------------------------------
        # [11] OPTIONAL LLM JUDGE (evaluation diagnostics only)
        # ------------------------------------------------------------
        if self.enable_judge:
            print("-" * 60)
            print("[11] LLM JUDGE (evaluation diagnostic, not ground truth)")
            print("-" * 60)
            print()
            self._print_llm_judge(normalized, (gt or {}).get("query_id", "DEMO"), systems)
            print()

        return reform_results

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
    parser.add_argument("--judge", action="store_true",
                        help="Show evaluation-only LLM judge grades (diagnostic; never ground truth)")
    args = parser.parse_args()

    demo = PresentationDemo(enable_ce=args.enable_ce, use_cache=args.use_cache, enable_judge=args.judge)
    if args.query:
        demo.execute_demo_query(args.query)
    else:
        demo.run()


if __name__ == "__main__":
    main()
