"""Phase 3 — End-to-End Enterprise Search Effectiveness Benchmark.

Executes:
1. System A: Conventional Enterprise Search Baseline
2. Configuration R0: Controlled Reformulation Ablation Baseline (No Reformulation)
3. System C / Configuration R1: Complete Proposed Query Reformulation Engine

Outputs:
- evaluation/phase3/phase3_system_a_answers.json
- evaluation/phase3/phase3_reformulation_ablation_answers.json
- evaluation/phase3/phase3_system_c_answers.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import dotenv
dotenv.load_dotenv()

from embedding.encoder import EmbeddingEncoder
from evaluation.metrics import calculate_all_metrics
from reformulation.answer_generator import AnswerGenerator
from reformulation.context_assembler import RAGContextAssembler
from reformulation.cross_encoder import CrossEncoderReranker
from reformulation.intent_router import IntentRouter
from reformulation.llm.fallback import FallbackLLMClient
from reformulation.normalizer import normalize_query
from reformulation.search_pipeline import SearchPipeline
from reformulation.semantic_similarity import SemanticSimilarity
from reformulation.spelling import CorpusSpellCorrector
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievedChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase3_benchmark")

# Paths
CHUNKS_PATH = os.path.join(ROOT, "Data", "chunks", "chunks.jsonl")
MATRIX_PATH = os.path.join(ROOT, "evaluation", "cache", "contextual_passage_matrix.npy")
REF_ANSWERS_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_reference_answers.json")
CACHE_REFORMULATIONS_PATH = os.path.join(ROOT, "evaluation", "cache", "heldout_queries_reformulations_gemini-3.5-flash-lite.json")
CACHE_DIR = os.path.join(ROOT, "evaluation", "phase3", "cache")
LLM_ANSWERS_CACHE_PATH = os.path.join(CACHE_DIR, "llm_generation_cache.json")

SYS_A_OUT = os.path.join(ROOT, "evaluation", "phase3", "phase3_system_a_answers.json")
SYS_C_OUT = os.path.join(ROOT, "evaluation", "phase3", "phase3_system_c_answers.json")
ABLATION_OUT = os.path.join(ROOT, "evaluation", "phase3", "phase3_reformulation_ablation_answers.json")


class CachedLocalVectorRetriever:
    """Exact cosine similarity vector retriever using pre-encoded passage matrix."""

    def __init__(self, chunks: List[dict], encoder: EmbeddingEncoder, matrix_path: str):
        self.chunks = chunks
        self.encoder = encoder
        self.matrix = np.load(matrix_path)
        self._query_cache: Dict[str, np.ndarray] = {}

    def retrieve(self, query: str, match_count: int = 20, filter_domain: Optional[str] = None, **_) -> List[RetrievedChunk]:
        if query not in self._query_cache:
            self._query_cache[query] = np.asarray(self.encoder.embed_query(query), dtype=np.float32)
        sims = self.matrix @ self._query_cache[query]
        order = np.argsort(-sims)
        out = []
        for i in order:
            c = self.chunks[i]
            if filter_domain and (c.get("department") or c.get("domain")) != filter_domain:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    title=c["title"],
                    domain=c.get("department") or c.get("domain") or "general",
                    chunk_index=c.get("chunk_index", 0),
                    chunk_text=c.get("text") or c.get("chunk_text") or "",
                    section=c.get("section"),
                    token_count=c.get("token_count"),
                    score=float(sims[i]),
                    retriever_type="vector",
                )
            )
            if len(out) >= match_count:
                break
        return out


class DictCachedReformulator:
    """Serves fixed reformulations from the evaluation cache keyed by query_id or query string."""

    def __init__(self, cache_dict: Dict[str, List[str]]):
        self.cache = cache_dict
        self.active_query_id: Optional[str] = None

    def generate_variants(self, query: str, max_variants: int = 5) -> List[str]:
        if self.active_query_id and self.active_query_id in self.cache:
            return list(self.cache[self.active_query_id])[:max_variants]
        return list(self.cache.get(query, []))[:max_variants]


class EmptyReformulator:
    def generate_variants(self, query: str, max_variants: int = 5) -> List[str]:
        return []


def load_llm_cache() -> Dict[str, dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(LLM_ANSWERS_CACHE_PATH):
        try:
            with open(LLM_ANSWERS_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_llm_cache(cache: Dict[str, dict]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(LLM_ANSWERS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def run_benchmark():
    logger.info("Initializing benchmark components...")
    with open(REF_ANSWERS_PATH, "r", encoding="utf-8") as f:
        ref_queries = json.load(f)
    logger.info("Loaded %d reference queries.", len(ref_queries))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d corpus chunks.", len(chunks))

    encoder = EmbeddingEncoder("BAAI/bge-small-en-v1.5")
    vec_retriever = CachedLocalVectorRetriever(chunks, encoder, MATRIX_PATH)
    bm25_retriever = BM25Retriever(chunks=chunks)
    hybrid_retriever = HybridRetriever(vector_retriever=vec_retriever, keyword_retriever=bm25_retriever)
    speller = CorpusSpellCorrector.from_chunks_file(CHUNKS_PATH)
    intent_router = IntentRouter.from_chunks_file(CHUNKS_PATH)
    sem_similarity = SemanticSimilarity(embed_fn=encoder.embed_query)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", enabled=True)

    with open(CACHE_REFORMULATIONS_PATH, "r", encoding="utf-8") as f:
        ref_cache_raw = json.load(f)
    reformulations_map = ref_cache_raw.get("reformulations", ref_cache_raw)
    cached_reformulator = DictCachedReformulator(reformulations_map)

    # Context assemblers
    legacy_assembler = RAGContextAssembler(strategy="legacy", max_total_chunks=5, max_tokens=2000)
    strategy_d_assembler = RAGContextAssembler(
        strategy="relevance_first",
        dominant_score_margin=5.2,
        lookahead_window=8,
        max_total_chunks=6,
        max_dominant_doc_chunks=4,
        max_chunks_per_doc=2,
        redundancy_threshold=0.70,
        max_tokens=2000,
    )

    # LLM Answer Generator
    llm_client = FallbackLLMClient.from_env()
    ans_gen_legacy = AnswerGenerator(llm_client, context_assembler=legacy_assembler)
    ans_gen_strat_d = AnswerGenerator(llm_client, context_assembler=strategy_d_assembler)

    # Search Pipelines
    # 1. System A: Conventional Baseline (no spelling, no reformulation, no reranking, equal RRF)
    pipe_a = SearchPipeline(
        reformulator=EmptyReformulator(),
        semantic_similarity=sem_similarity,
        hybrid_retriever=hybrid_retriever,
        spell_corrector=None,
        intent_router=None,
        multi_dense_single_bm25=False,
        reranker=None,
    )

    # 2. Config R0: Controlled Ablation (spelling, intent routing, no reformulation, contextual cross-encoder, Strategy D)
    pipe_r0 = SearchPipeline(
        reformulator=EmptyReformulator(),
        semantic_similarity=sem_similarity,
        hybrid_retriever=hybrid_retriever,
        spell_corrector=speller,
        intent_router=intent_router,
        multi_dense_single_bm25=False,
        reranker=reranker,
        rerank_top_k=15,
    )

    # 3. System C / Config R1: Complete Proposed Engine (spelling, intent routing, multi-dense reformulation, reranker, Strategy D)
    pipe_c = SearchPipeline(
        reformulator=cached_reformulator,
        semantic_similarity=sem_similarity,
        hybrid_retriever=hybrid_retriever,
        spell_corrector=speller,
        intent_router=intent_router,
        multi_dense_single_bm25=True,
        fusion_strategy="weighted",
        reranker=reranker,
        rerank_top_k=15,
    )

    llm_cache = load_llm_cache()
    logger.info("Loaded LLM answer cache with %d entries.", len(llm_cache))

    sys_a_results = []
    config_r0_results = []
    sys_c_results = []

    total_queries = len(ref_queries)
    t_start = time.time()

    for idx, q_item in enumerate(ref_queries, start=1):
        qid = q_item["query_id"]
        raw_query = q_item["original_query"]
        cached_reformulator.active_query_id = qid

        # ---------------------------------------------------------------------
        # 1. System A Execution
        # ---------------------------------------------------------------------
        search_res_a = pipe_a.search_detailed(
            raw_query,
            candidate_k=20,
            final_k=20,
            primary_query_weight=1.0,
        )
        ctx_a = legacy_assembler.assemble_context(raw_query, search_res_a.results)
        prompt_a = ans_gen_legacy.build_prompt(raw_query, search_res_a.results)

        cache_key_a = f"A::{qid}::{raw_query}"
        if cache_key_a in llm_cache:
            ans_payload_a = llm_cache[cache_key_a]
        else:
            try:
                ans_obj_a = ans_gen_legacy.generate(raw_query, search_res_a.results)
                ans_payload_a = ans_obj_a.to_dict()
                ans_payload_a["error"] = None
            except Exception as e:
                logger.error("System A LLM generation failed for %s: %s", qid, e)
                ans_payload_a = {
                    "text": "The available documents do not answer this question.",
                    "citations": [],
                    "answerable": False,
                    "cited_documents": [],
                    "missing_aspects": [],
                    "context_token_count": 0,
                    "error": str(e),
                }
            llm_cache[cache_key_a] = ans_payload_a
            save_llm_cache(llm_cache)

        record_a = {
            "query_id": qid,
            "original_query": raw_query,
            "normalized_query": search_res_a.normalized_query,
            "retrieved_chunk_ids": [c.chunk_id for c in search_res_a.results],
            "context_chunk_ids": [c.chunk_id for c in ctx_a.chunks],
            "context_text": ctx_a.formatted_sources,
            "context_token_count": ctx_a.total_tokens,
            "prompt": prompt_a,
            "answer": ans_payload_a,
            "timings_ms": search_res_a.timings_ms,
        }
        sys_a_results.append(record_a)

        # ---------------------------------------------------------------------
        # 2. Configuration R0 Execution (Ablation: No Reformulation, Cross-Encoder + Strategy D)
        # ---------------------------------------------------------------------
        search_res_r0 = pipe_r0.search_detailed(
            raw_query,
            primary_query_weight=1.5,
            candidate_k=50,
            final_k=20,
        )
        ctx_r0 = strategy_d_assembler.assemble_context(raw_query, search_res_r0.results)
        prompt_r0 = ans_gen_strat_d.build_prompt(raw_query, search_res_r0.results)

        cache_key_r0 = f"R0::{qid}::{raw_query}"
        if cache_key_r0 in llm_cache:
            ans_payload_r0 = llm_cache[cache_key_r0]
        else:
            try:
                ans_obj_r0 = ans_gen_strat_d.generate(raw_query, search_res_r0.results)
                ans_payload_r0 = ans_obj_r0.to_dict()
                ans_payload_r0["error"] = None
            except Exception as e:
                logger.error("Config R0 LLM generation failed for %s: %s", qid, e)
                ans_payload_r0 = {
                    "text": "The available documents do not answer this question.",
                    "citations": [],
                    "answerable": False,
                    "cited_documents": [],
                    "missing_aspects": [],
                    "context_token_count": 0,
                    "error": str(e),
                }
            llm_cache[cache_key_r0] = ans_payload_r0
            save_llm_cache(llm_cache)

        record_r0 = {
            "query_id": qid,
            "original_query": raw_query,
            "normalized_query": search_res_r0.normalized_query,
            "retrieved_chunk_ids": [c.chunk_id for c in search_res_r0.results],
            "context_chunk_ids": [c.chunk_id for c in ctx_r0.chunks],
            "context_text": ctx_r0.formatted_sources,
            "context_token_count": ctx_r0.total_tokens,
            "prompt": prompt_r0,
            "answer": ans_payload_r0,
            "timings_ms": search_res_r0.timings_ms,
        }
        config_r0_results.append(record_r0)

        # ---------------------------------------------------------------------
        # 3. System C / Configuration R1 Execution (Complete Proposed Engine)
        # ---------------------------------------------------------------------
        search_res_c = pipe_c.search_detailed(
            raw_query,
            primary_query_weight=1.5,
            candidate_k=50,
            final_k=20,
        )
        ctx_c = strategy_d_assembler.assemble_context(raw_query, search_res_c.results)
        prompt_c = ans_gen_strat_d.build_prompt(raw_query, search_res_c.results)

        cache_key_c = f"C::{qid}::{raw_query}"
        if cache_key_c in llm_cache:
            ans_payload_c = llm_cache[cache_key_c]
        elif prompt_c == prompt_r0 and ans_payload_r0.get("error") is None:
            # Identical prompt between R0 and C (e.g. for DIRECT search queries where no reformulations were run)
            ans_payload_c = dict(ans_payload_r0)
            llm_cache[cache_key_c] = ans_payload_c
            save_llm_cache(llm_cache)
        else:
            try:
                ans_obj_c = ans_gen_strat_d.generate(raw_query, search_res_c.results)
                ans_payload_c = ans_obj_c.to_dict()
                ans_payload_c["error"] = None
            except Exception as e:
                logger.error("System C LLM generation failed for %s: %s", qid, e)
                ans_payload_c = {
                    "text": "The available documents do not answer this question.",
                    "citations": [],
                    "answerable": False,
                    "cited_documents": [],
                    "missing_aspects": [],
                    "context_token_count": 0,
                    "error": str(e),
                }
            llm_cache[cache_key_c] = ans_payload_c
            save_llm_cache(llm_cache)

        record_c = {
            "query_id": qid,
            "original_query": raw_query,
            "normalized_query": search_res_c.normalized_query,
            "reformulations": search_res_c.reformulations,
            "queries_used": search_res_c.queries_used,
            "detected_intent": search_res_c.detected_intent,
            "routing_decision": search_res_c.routing_decision,
            "retrieved_chunk_ids": [c.chunk_id for c in search_res_c.results],
            "context_chunk_ids": [c.chunk_id for c in ctx_c.chunks],
            "context_text": ctx_c.formatted_sources,
            "context_token_count": ctx_c.total_tokens,
            "prompt": prompt_c,
            "answer": ans_payload_c,
            "timings_ms": search_res_c.timings_ms,
        }
        sys_c_results.append(record_c)

        if idx % 10 == 0 or idx == total_queries:
            elapsed = time.time() - t_start
            logger.info("Processed %d/%d queries (%.1fs elapsed).", idx, total_queries, elapsed)

    # Save deliverables
    logger.info("Writing results to disk...")
    with open(SYS_A_OUT, "w", encoding="utf-8") as f:
        json.dump(sys_a_results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote System A outputs: %s", SYS_A_OUT)

    with open(SYS_C_OUT, "w", encoding="utf-8") as f:
        json.dump(sys_c_results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote System C outputs: %s", SYS_C_OUT)

    ablation_payload = {
        "configuration_r0": config_r0_results,
        "configuration_r1": sys_c_results,
    }
    with open(ABLATION_OUT, "w", encoding="utf-8") as f:
        json.dump(ablation_payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote Reformulation Ablation outputs: %s", ABLATION_OUT)

    logger.info("Phase 3 benchmark answer generation completed in %.2fs", time.time() - t_start)


if __name__ == "__main__":
    run_benchmark()
