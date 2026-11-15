"""Validation script for System A, Configuration R0, and System C execution."""

import json
import os
import sys
import dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
dotenv.load_dotenv()

import numpy as np
from embedding.encoder import EmbeddingEncoder
from reformulation.cross_encoder import CrossEncoderReranker
from reformulation.intent_router import IntentRouter
from reformulation.normalizer import normalize_query
from reformulation.semantic_similarity import SemanticSimilarity
from reformulation.spelling import CorpusSpellCorrector
from reformulation.search_pipeline import SearchPipeline
from reformulation.context_assembler import RAGContextAssembler
from reformulation.answer_generator import AnswerGenerator
from reformulation.llm.fallback import FallbackLLMClient
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievedChunk

CHUNKS_PATH = os.path.join(ROOT, "Data", "chunks", "chunks.jsonl")
MATRIX_PATH = os.path.join(ROOT, "evaluation", "cache", "contextual_passage_matrix.npy")
CACHE_REFORMULATIONS = os.path.join(ROOT, "evaluation", "cache", "heldout_queries_reformulations_gemini-3.5-flash-lite.json")

class CachedLocalVectorRetriever:
    def __init__(self, chunks, encoder, matrix_path):
        self.chunks = chunks
        self.encoder = encoder
        self.matrix = np.load(matrix_path)
        self._query_cache = {}

    def retrieve(self, query: str, match_count: int = 20, filter_domain = None, **_):
        if query not in self._query_cache:
            self._query_cache[query] = np.asarray(self.encoder.embed_query(query), dtype=np.float32)
        sims = self.matrix @ self._query_cache[query]
        order = np.argsort(-sims)
        out = []
        for i in order:
            c = self.chunks[i]
            if filter_domain and c.get("domain") != filter_domain:
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
                )
            )
            if len(out) >= match_count:
                break
        return out

class DictCachedReformulator:
    def __init__(self, cache_dict):
        self.cache = cache_dict

    def generate_variants(self, query: str, max_variants: int = 5):
        # lookup by query string
        return self.cache.get(query, [])

def main():
    print("Loading dependencies...")
    chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
    encoder = EmbeddingEncoder("BAAI/bge-small-en-v1.5")
    vec = CachedLocalVectorRetriever(chunks, encoder, MATRIX_PATH)
    bm25 = BM25Retriever(chunks=chunks)
    hybrid = HybridRetriever(vector_retriever=vec, keyword_retriever=bm25)
    speller = CorpusSpellCorrector.from_chunks_file(CHUNKS_PATH)
    intent_router = IntentRouter.from_chunks_file(CHUNKS_PATH)
    sem = SemanticSimilarity(embed_fn=encoder.embed_query)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", enabled=True)

    with open(CACHE_REFORMULATIONS, encoding="utf-8") as f:
        ref_data = json.load(f)
    reformulations_map = ref_data.get("reformulations", ref_data)
    cached_ref = DictCachedReformulator(reformulations_map)

    # Context Assemblers
    legacy_assembler = RAGContextAssembler(strategy="legacy", max_total_chunks=5)
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

    test_q = "bus pass refund"
    print(f"Testing Query: '{test_q}'")

    # 1. System A
    pipe_a = SearchPipeline(
        reformulator=DictCachedReformulator({}),
        semantic_similarity=sem,
        hybrid_retriever=hybrid,
        spell_corrector=None,
        intent_router=None,
        multi_dense_single_bm25=False,
        reranker=None,
    )
    res_a = pipe_a.search_detailed(test_q, candidate_k=20, final_k=20).results
    ctx_a = legacy_assembler.assemble_context(test_q, res_a)
    print(f"System A Context: {len(ctx_a.chunks)} chunks, Top chunk: {ctx_a.chunks[0].chunk_id}")

    # 2. Config R0
    pipe_r0 = SearchPipeline(
        reformulator=DictCachedReformulator({}),
        semantic_similarity=sem,
        hybrid_retriever=hybrid,
        spell_corrector=speller,
        intent_router=intent_router,
        multi_dense_single_bm25=False,
        reranker=reranker,
        rerank_top_k=15,
    )
    res_r0 = pipe_r0.search_detailed(test_q, primary_query_weight=1.5, candidate_k=50, final_k=20).results
    ctx_r0 = strategy_d_assembler.assemble_context(test_q, res_r0)
    print(f"Config R0 Context: {len(ctx_r0.chunks)} chunks, Top chunk: {ctx_r0.chunks[0].chunk_id}")

    # 3. System C / R1
    pipe_c = SearchPipeline(
        reformulator=cached_ref,
        semantic_similarity=sem,
        hybrid_retriever=hybrid,
        spell_corrector=speller,
        intent_router=intent_router,
        multi_dense_single_bm25=True,
        fusion_strategy="weighted",
        reranker=reranker,
        rerank_top_k=15,
    )
    res_c = pipe_c.search_detailed(test_q, primary_query_weight=1.5, candidate_k=50, final_k=20).results
    ctx_c = strategy_d_assembler.assemble_context(test_q, res_c)
    print(f"System C Context: {len(ctx_c.chunks)} chunks, Top chunk: {ctx_c.chunks[0].chunk_id}")

    print("All three pipelines validated successfully!")

if __name__ == "__main__":
    main()
