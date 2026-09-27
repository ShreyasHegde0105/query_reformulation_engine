"""Phase 5 — Candidate Generation evaluation.

Compares the Phase 2 retrieval configuration against the Phase 5 changes:

* **Contextual passages** — chunks embedded as ``title / section / text``
  (``DocumentMapper.passage_text``) instead of bare chunk text.
* **BM25 keyword branch** — in-process Okapi BM25 (``BM25Retriever``) over
  title + section + text instead of PostgreSQL ``websearch_to_tsquery`` +
  ``ts_rank_cd`` (AND semantics, no IDF).

Vector search is run in-process as exact cosine similarity over locally
encoded chunks, so both passage templates can be compared without re-ingesting
Supabase. (With plain passages this reproduces the Supabase ``match_chunks_vector``
results for the Phase 2 baseline.) Reformulations come from the fixed cache in
``evaluation/cache/reformulations.json``, so the run is deterministic and
needs no LLM.

Usage::

    python evaluation/run_phase5_evaluation.py                     # fully offline
    python evaluation/run_phase5_evaluation.py --postgres-keyword  # + Phase 2 reference (read-only Supabase RPC)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding.encoder import EmbeddingEncoder
from embedding.mapper import DocumentMapper
from evaluation.metrics import calculate_all_metrics
from reformulation.search_pipeline import SearchPipeline
from reformulation.semantic_similarity import SemanticSimilarity
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievedChunk

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluation.phase5")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKS_PATH = os.path.join(ROOT, "Data", "chunks", "chunks.jsonl")
GOLDEN_PATH = os.path.join(ROOT, "evaluation", "golden_queries.json")
CACHE_PATH = os.path.join(ROOT, "evaluation", "cache", "reformulations.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

FINAL_K = 20          # same output depth as the Phase 2 benchmark
POOL_K = 50           # candidate-pool depth (Phase 4 comparison)
MATCH_COUNT = 20      # per-branch candidates per query
RRF_K = 60
MAX_VARIANTS = 5


class LocalVectorRetriever:
    """Exact cosine search over locally encoded chunks (VectorRetriever contract)."""

    def __init__(self, chunks: List[dict], encoder: EmbeddingEncoder, contextual: bool) -> None:
        self.chunks = chunks
        self.encoder = encoder
        passages = [DocumentMapper.passage_text(c, contextual) for c in chunks]
        self.matrix = np.asarray(encoder.encode_passages(passages, show_progress=False), dtype=np.float32)
        self._query_cache: Dict[str, np.ndarray] = {}

    def retrieve(self, query: str, match_count: int = 20, filter_domain: Optional[str] = None, **_) -> List[RetrievedChunk]:
        if query not in self._query_cache:
            self._query_cache[query] = np.asarray(self.encoder.embed_query(query), dtype=np.float32)
        sims = self.matrix @ self._query_cache[query]
        order = np.argsort(-sims)
        out: List[RetrievedChunk] = []
        for i in order:
            c = self.chunks[i]
            if filter_domain and c.get("department") != filter_domain:
                continue
            out.append(RetrievedChunk(
                chunk_id=c["chunk_id"], document_id=c["document_id"], title=c["title"],
                domain=c.get("department") or "", chunk_index=c["chunk_index"], chunk_text=c["text"],
                section=c.get("section"), token_count=c.get("token_count"), score=float(sims[i]),
                retriever_type="vector",
            ))
            if len(out) >= match_count:
                break
        return out


class CachedReformulator:
    """Serves fixed reformulations for the query currently being evaluated."""

    def __init__(self, cache: Dict[str, List[str]]) -> None:
        self.cache = cache
        self.query_id: Optional[str] = None

    def generate_variants(self, query: str, max_variants: int = 5) -> List[str]:
        return self.cache.get(self.query_id or "", [])[:max_variants]


def evaluate(golden: List[dict], run_fn: Callable[[dict, int], List[str]], reformulator: CachedReformulator) -> dict:
    """Run one configuration over the golden set at FINAL_K and POOL_K depth."""
    per_query = []
    for g in golden:
        reformulator.query_id = g["query_id"]
        ranked = run_fn(g, FINAL_K)
        pool = run_fn(g, POOL_K)
        m = calculate_all_metrics(ranked, g["relevant_chunks"], g.get("graded_relevance"), k_list=[5, 10])
        rel = set(g["relevant_chunks"])
        m["recall@20"] = len(rel & set(ranked[:20])) / len(rel)
        m["pool_recall@50"] = len(rel & set(pool[:POOL_K])) / len(rel)
        per_query.append({"query_id": g["query_id"], "query": g["query"], "category": g["category"],
                          "retrieved": ranked, "metrics": m})

    keys = per_query[0]["metrics"].keys()
    mean = {k: float(np.mean([q["metrics"][k] for q in per_query])) for k in keys}
    categories: Dict[str, Dict[str, float]] = {}
    for cat in sorted({q["category"] for q in per_query}):
        rows = [q for q in per_query if q["category"] == cat]
        categories[cat] = {k: float(np.mean([q["metrics"][k] for q in rows])) for k in keys}
    return {"mean": mean, "categories": categories, "per_query": per_query}


def build_runners(chunks, encoder, sem, reformulator, postgres_keyword: bool):
    vec_plain = LocalVectorRetriever(chunks, encoder, contextual=False)
    vec_ctx = LocalVectorRetriever(chunks, encoder, contextual=True)
    bm25 = BM25Retriever(chunks=chunks)

    def single(retriever):
        return lambda g, depth: [c.chunk_id for c in retriever.retrieve(g["query"], match_count=depth)]

    def hybrid(vec, kw):
        h = HybridRetriever(vector_retriever=vec, keyword_retriever=kw)
        return lambda g, depth: [c.chunk_id for c in h.retrieve(
            g["query"], vector_match_count=MATCH_COUNT, keyword_match_count=MATCH_COUNT, k=RRF_K, final_k=depth)]

    def pipeline(vec, kw):
        p = SearchPipeline(reformulator=reformulator, semantic_similarity=sem,
                           hybrid_retriever=HybridRetriever(vector_retriever=vec, keyword_retriever=kw))
        return lambda g, depth: [c.chunk_id for c in p.search(
            g["query"], max_variants=MAX_VARIANTS, vector_match_count=MATCH_COUNT,
            keyword_match_count=MATCH_COUNT, k=RRF_K, final_k=depth)]

    runners: Dict[str, tuple[str, Callable]] = {}
    if postgres_keyword:
        from retrieval.keyword_retriever import KeywordRetriever
        pg = KeywordRetriever()
        runners["p2_keyword_pg"] = ("Postgres FTS (Phase 2 keyword)", single(pg))
        runners["p2_pipeline"] = ("Phase 2 pipeline: plain vec + Postgres FTS", pipeline(vec_plain, pg))
    runners.update({
        "vector_plain": ("Vector, plain passages", single(vec_plain)),
        "vector_ctx": ("Vector, contextual passages", single(vec_ctx)),
        "bm25": ("BM25 (title+section+text)", single(bm25)),
        "hybrid_ctx_bm25": ("Hybrid: contextual vec + BM25 (no reformulation)", hybrid(vec_ctx, bm25)),
        "pipeline_plain_bm25": ("Pipeline: plain vec + BM25", pipeline(vec_plain, bm25)),
        "pipeline_ctx_bm25": ("Pipeline: contextual vec + BM25 (Phase 5)", pipeline(vec_ctx, bm25)),
    })
    return runners


def write_report(results: Dict[str, dict], names: Dict[str, str], path: str, baseline_key: str, final_key: str) -> None:
    fmt = lambda x: f"{x:.4f}"
    lines = [
        "# Phase 5 Report — Candidate Generation",
        "",
        f"**Run:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Queries:** 50 golden queries · **Output depth:** top-{FINAL_K} · **Pool depth:** top-{POOL_K} · "
        f"**Per-branch candidates:** {MATCH_COUNT} · **RRF k:** {RRF_K}  ",
        "**Reformulations:** fixed cache (`evaluation/cache/reformulations.json`) · "
        "**Vector search:** exact cosine, in-process (bge-small-en-v1.5)",
        "",
        "## 1. Overall",
        "",
        "| Configuration | R@5 | R@10 | R@20 | MRR | NDCG@5 | NDCG@10 | Pool R@50 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, res in results.items():
        m = res["mean"]
        lines.append(f"| {names[key]} | {fmt(m['recall@5'])} | {fmt(m['recall@10'])} | {fmt(m['recall@20'])} | "
                     f"{fmt(m['mrr'])} | {fmt(m['ndcg@5'])} | {fmt(m['ndcg@10'])} | {fmt(m['pool_recall@50'])} |")

    base, final = results[baseline_key], results[final_key]
    lines += [
        "",
        f"## 2. Per-category: {names[baseline_key]} → {names[final_key]}",
        "",
        "| Category | R@10 before | R@10 after | MRR before | MRR after | NDCG@10 before | NDCG@10 after |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for cat, bm in base["categories"].items():
        fm = final["categories"][cat]
        lines.append(f"| `{cat}` | {fmt(bm['recall@10'])} | {fmt(fm['recall@10'])} | {fmt(bm['mrr'])} | "
                     f"{fmt(fm['mrr'])} | {fmt(bm['ndcg@10'])} | {fmt(fm['ndcg@10'])} |")

    base_q = {q["query_id"]: q for q in base["per_query"]}
    improved, regressed = [], []
    for q in final["per_query"]:
        b = base_q[q["query_id"]]["metrics"]
        d_mrr, d_r10 = q["metrics"]["mrr"] - b["mrr"], q["metrics"]["recall@10"] - b["recall@10"]
        row = (q["query_id"], q["category"], q["query"], b["mrr"], q["metrics"]["mrr"], b["recall@10"], q["metrics"]["recall@10"])
        if d_mrr > 1e-9 or d_r10 > 1e-9:
            if d_mrr >= -1e-9 and d_r10 >= -1e-9:
                improved.append(row)
            else:
                regressed.append(row)
        elif d_mrr < -1e-9 or d_r10 < -1e-9:
            regressed.append(row)

    for title, rows in (("3. Improved queries", improved), ("4. Regressed or mixed queries", regressed)):
        lines += ["", f"## {title} ({len(rows)})", "",
                  "| Query | Category | Text | MRR before | MRR after | R@10 before | R@10 after |",
                  "|:---|:---|:---|---:|---:|---:|---:|"]
        lines += [f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.2f} | {r[6]:.2f} |" for r in rows]

    lines += [
        "",
        "## 5. Notes",
        "",
        "- With 50 queries, one query moves Recall@10 by 0.01–0.02. Treat differences below ~0.03 as noise.",
        "- BM25 indexes title, section and chunk text only; the synthetic `keywords` metadata is deliberately "
        "excluded, because it was produced by the same generator as the golden queries.",
        "- Contextual passages change only the embedded text; the stored `chunk_text` is unchanged. Production "
        "vector search picks this up after re-running ingestion (`EMBEDDING_CONTEXTUAL=1`, the default).",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 candidate-generation evaluation")
    parser.add_argument("--postgres-keyword", action="store_true",
                        help="Also run the Phase 2 reference using the live Supabase keyword RPC (read-only).")
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)
    with open(CACHE_PATH, encoding="utf-8") as f:
        reformulator = CachedReformulator(json.load(f))

    encoder = EmbeddingEncoder()
    sem = SemanticSimilarity(embed_fn=encoder.embed_query)
    runners = build_runners(chunks, encoder, sem, reformulator, args.postgres_keyword)

    results: Dict[str, dict] = {}
    for key, (name, fn) in runners.items():
        print(f"Evaluating: {name} ...", flush=True)
        results[key] = evaluate(golden, fn, reformulator)
        m = results[key]["mean"]
        print(f"    R@10={m['recall@10']:.3f}  MRR={m['mrr']:.3f}  NDCG@10={m['ndcg@10']:.3f}  "
              f"Pool@50={m['pool_recall@50']:.3f}", flush=True)

    names = {k: v[0] for k, v in runners.items()}
    baseline_key = "p2_pipeline" if "p2_pipeline" in results else "pipeline_plain_bm25"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "phase5_results.json"), "w", encoding="utf-8") as f:
        json.dump({"names": names, "results": results}, f, indent=1)
    write_report(results, names, os.path.join(args.output_dir, "phase5_report.md"), baseline_key, "pipeline_ctx_bm25")
    print(f"\nReport: {os.path.join(args.output_dir, 'phase5_report.md')}")


if __name__ == "__main__":
    main()
