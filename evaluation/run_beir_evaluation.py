"""External validation on BEIR FiQA-2018 (57,638 passages, 648 test queries).

Does LLM query reformulation beat normal search, and cheap non-LLM expansion,
on data this project did not create? Compares, at identical retrieval depth:

    bm25         BM25 (in-process, IDF-weighted)
    dense        bge-small-en-v1.5, exact cosine
    hybrid       dense + BM25, RRF
    hybrid_rm3   dense + RM3-expanded BM25, RRF          (non-LLM expansion)
    llm_pipeline full SearchPipeline: normalize → LLM rewrites → filter →
                 multi-query hybrid RRF                   (this project)

Three steps (the LLM step is rate-limited; encoding is the CPU/memory-heavy one):

    python evaluation/run_beir_evaluation.py encode        # dense index, cached (run once)
    python evaluation/run_beir_evaluation.py reformulate   # resumable; stops at the daily quota
    python evaluation/run_beir_evaluation.py evaluate      # all systems + significance tests

Data: download https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
and unzip into Data/beir/ (gitignored).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import calculate_all_metrics
from evaluation.significance import paired_comparison
from retrieval.models import RetrievedChunk

logger = logging.getLogger("evaluation.beir")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "Data", "beir", "fiqa")
DENSE_CACHE = os.path.join(DATA_DIR, "bge_small_passages.npz")
CACHE_DIR = os.path.join(ROOT, "evaluation", "cache")


def cache_path_for(model: str) -> str:
    """One reformulation cache per LLM, so rewrites from different models never mix."""
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model.split("/")[-1])
    return os.path.join(CACHE_DIR, f"beir_fiqa_reformulations_{safe}.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

MATCH_COUNT = 50      # per-branch candidates (same for every system)
FINAL_K = 20          # ranked output depth used for metrics
MAX_VARIANTS = 5
RRF_K = 60

SYSTEMS = {
    "bm25": "BM25",
    "dense": "Dense (bge-small)",
    "hybrid": "Hybrid (dense + BM25, RRF)",
    "hybrid_rm3": "Hybrid + RM3 expansion (non-LLM)",
    "llm_pipeline": "Hybrid + LLM reformulation (full pipeline)",
}
COMPARISONS = [
    ("hybrid", "llm_pipeline"),
    ("hybrid_rm3", "llm_pipeline"),
    ("hybrid", "hybrid_rm3"),
    ("bm25", "hybrid"),
    ("dense", "hybrid"),
]
METRICS = ("ndcg@10", "recall@10", "mrr", "recall@20")


@contextlib.contextmanager
def keep_awake():
    """Ask Windows not to idle-sleep while a long benchmark step runs.

    Per-process and released on exit (it does not change power settings); closing
    the lid still sleeps. A no-op on other platforms.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_fiqa(data_dir: str = DATA_DIR, split: str = "test"):
    """Return (chunks, queries, qrels) for one split.

    chunks: chunk dicts in this project's format (one per passage)
    queries: {query_id: text} for queries that have relevance labels in ``split``
    qrels: {query_id: {doc_id: relevance}}
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with open(os.path.join(data_dir, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)

    queries: Dict[str, str] = {}
    with open(os.path.join(data_dir, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries[q["_id"]] = q["text"]

    chunks: List[Dict[str, Any]] = []
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            chunks.append({
                "chunk_id": d["_id"], "document_id": d["_id"], "title": d.get("title") or "",
                "section": None, "text": d["text"], "chunk_index": 0, "department": "fiqa",
            })
    return chunks, queries, qrels


# ----------------------------------------------------------------------------
# Step 1: reformulations (rate-limited, resumable)
# ----------------------------------------------------------------------------
def load_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"reformulations": {}, "parse_errors": {}}


def save_cache(cache: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def generate_reformulations(
    queries: Dict[str, str],
    reformulator: Any,
    cache_path: str,
    min_interval: float = 13.0,
    rate_limit_wait: float = 65.0,
    max_consecutive_rate_limits: int = 3,
    max_consecutive_errors: int = 5,
    limit: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fill the reformulation cache for every query not yet in it.

    * Paces calls ``min_interval`` seconds apart (13s ≈ 4.6 requests/min, under a
      5/min free tier).
    * On a rate limit, waits ``rate_limit_wait`` and retries the same query; after
      ``max_consecutive_rate_limits`` in a row the daily quota is assumed spent and
      the run stops (re-run later to resume).
    * An unparseable LLM reply is cached as ``[]`` (the pipeline would fall back to
      the original query) and recorded under ``parse_errors``.
    * The cache is written after every query, so interruption loses nothing.
    """
    from reformulation.exceptions import ReformulationParseError
    from reformulation.llm.exceptions import LLMRateLimitError

    cache = load_cache(cache_path)
    cache.setdefault("reformulations", {})
    cache.setdefault("parse_errors", {})
    if metadata:
        cache.setdefault("metadata", metadata)
    done = cache["reformulations"]
    pending = [qid for qid in queries if qid not in done]
    if limit is not None:
        pending = pending[:limit]

    stats = {"already_cached": len(done), "new": 0, "parse_errors": 0, "stopped": None}
    last_call: Optional[float] = None
    rate_limits = errors = 0
    i = 0
    while i < len(pending):
        qid = pending[i]
        if last_call is not None:
            wait = min_interval - (clock() - last_call)
            if wait > 0:
                sleep(wait)
        last_call = clock()
        try:
            variants = reformulator.generate_variants(queries[qid], max_variants=MAX_VARIANTS)
        except LLMRateLimitError:
            rate_limits += 1
            if rate_limits >= max_consecutive_rate_limits:
                stats["stopped"] = "rate_limit (daily quota likely exhausted; re-run later to resume)"
                break
            logger.warning("Rate limited; waiting %.0fs (%d/%d)", rate_limit_wait, rate_limits, max_consecutive_rate_limits)
            sleep(rate_limit_wait)
            continue
        except ReformulationParseError as exc:
            variants = []
            cache["parse_errors"][qid] = str(exc)
            stats["parse_errors"] += 1
        except ValueError as exc:  # query rejected by validation
            variants = []
            cache["parse_errors"][qid] = f"validation: {exc}"
            stats["parse_errors"] += 1
        except Exception as exc:
            errors += 1
            logger.warning("Reformulation failed for %s: %s", qid, exc)
            if errors >= max_consecutive_errors:
                stats["stopped"] = f"errors ({type(exc).__name__}: {exc})"
                break
            i += 1
            continue

        rate_limits = errors = 0
        done[qid] = variants
        stats["new"] += 1
        save_cache(cache, cache_path)
        i += 1
        if stats["new"] % 10 == 0:
            print(f"  {len(done)}/{len(queries)} queries reformulated", flush=True)

    stats["total_cached"] = len(done)
    stats["remaining"] = len([q for q in queries if q not in done])
    return stats


# ----------------------------------------------------------------------------
# Step 2: evaluation
# ----------------------------------------------------------------------------
class DenseIndex:
    """Exact cosine search over pre-encoded passages (VectorRetriever contract)."""

    def __init__(self, chunks: List[Dict[str, Any]], encoder: Any, cache_path: Optional[str] = None) -> None:
        from embedding.mapper import DocumentMapper

        self.chunks = chunks
        self.encoder = encoder
        ids_hash = hashlib.sha1("\n".join(c["chunk_id"] for c in chunks).encode()).hexdigest()
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            if str(data["ids_hash"]) == ids_hash:
                self.matrix = data["matrix"]
                return
        passages = [DocumentMapper.passage_text(c) for c in chunks]
        self.matrix = np.asarray(encoder.encode_passages(passages, batch_size=64, show_progress=True), dtype=np.float32)
        if cache_path:
            np.savez(cache_path, matrix=self.matrix, ids_hash=ids_hash)

    def retrieve(self, query: str, match_count: int = 20, filter_domain: Optional[str] = None, **_) -> List[RetrievedChunk]:
        sims = self.matrix @ np.asarray(self.encoder.embed_query(query), dtype=np.float32)
        top = np.argpartition(-sims, min(match_count, len(sims) - 1))[:match_count]
        top = top[np.argsort(-sims[top])]
        return [
            RetrievedChunk(chunk_id=self.chunks[i]["chunk_id"], document_id=self.chunks[i]["document_id"],
                           title=self.chunks[i]["title"], domain="fiqa", chunk_index=0,
                           chunk_text=self.chunks[i]["text"], section=None, token_count=None,
                           score=float(sims[i]), retriever_type="vector")
            for i in top
        ]


class _CachedReformulator:
    def __init__(self, cache: Dict[str, List[str]], text_to_qid: Dict[str, str]) -> None:
        self.cache = cache
        self.text_to_qid = text_to_qid

    def generate_variants(self, query: str, max_variants: int = 5) -> List[str]:
        return self.cache.get(self.text_to_qid.get(query, ""), [])[:max_variants]


def build_systems(chunks, queries, reformulations, encoder) -> Dict[str, Callable[[str], List[str]]]:
    from reformulation.search_pipeline import SearchPipeline
    from reformulation.semantic_similarity import SemanticSimilarity
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.prf import RM3Retriever

    print("Building BM25 index ...", flush=True)
    bm25 = BM25Retriever(chunks=chunks)
    print("Loading / encoding dense index ...", flush=True)
    dense = DenseIndex(chunks, encoder, cache_path=DENSE_CACHE)
    hybrid = HybridRetriever(vector_retriever=dense, keyword_retriever=bm25)
    hybrid_rm3 = HybridRetriever(vector_retriever=dense, keyword_retriever=RM3Retriever(bm25))
    pipeline = SearchPipeline(
        reformulator=_CachedReformulator(reformulations, {t: q for q, t in queries.items()}),
        semantic_similarity=SemanticSimilarity(embed_fn=encoder.embed_query),
        hybrid_retriever=hybrid,
    )

    def ids(results):
        return [c.chunk_id for c in results]

    return {
        "bm25": lambda q: ids(bm25.retrieve(q, match_count=FINAL_K)),
        "dense": lambda q: ids(dense.retrieve(q, match_count=FINAL_K)),
        "hybrid": lambda q: ids(hybrid.retrieve(q, vector_match_count=MATCH_COUNT, keyword_match_count=MATCH_COUNT,
                                                k=RRF_K, final_k=FINAL_K)),
        "hybrid_rm3": lambda q: ids(hybrid_rm3.retrieve(q, vector_match_count=MATCH_COUNT, keyword_match_count=MATCH_COUNT,
                                                        k=RRF_K, final_k=FINAL_K)),
        "llm_pipeline": lambda q: ids(pipeline.search(q, max_variants=MAX_VARIANTS, vector_match_count=MATCH_COUNT,
                                                      keyword_match_count=MATCH_COUNT, k=RRF_K, final_k=FINAL_K,
                                                      candidate_k=MATCH_COUNT)),
    }


def score_query(ranked: List[str], relevant: Dict[str, int]) -> Dict[str, float]:
    m = calculate_all_metrics(ranked, list(relevant), relevant, k_list=[10])
    m["recall@20"] = len(set(ranked[:20]) & set(relevant)) / len(relevant)
    return m


def evaluate(queries, qrels, systems, query_ids) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for key, run in systems.items():
        print(f"Evaluating {SYSTEMS[key]} on {len(query_ids)} queries ...", flush=True)
        t0 = time.perf_counter()
        per_query, failures = {}, 0
        for qid in query_ids:
            try:
                ranked = run(queries[qid])
            except ValueError:
                ranked, failures = [], failures + 1   # query rejected by validation
            per_query[qid] = score_query(ranked, qrels[qid])
        mean = {m: float(np.mean([pq[m] for pq in per_query.values()])) for m in METRICS}
        results[key] = {"mean": mean, "per_query": per_query, "failures": failures,
                        "seconds_per_query": (time.perf_counter() - t0) / len(query_ids)}
        print("    " + "  ".join(f"{m}={mean[m]:.4f}" for m in METRICS), flush=True)
    return results


def write_report(results, query_ids, n_total, cache, cache_path, path) -> None:
    lines = [
        "# BEIR FiQA-2018 — Does LLM Query Reformulation Beat Normal Search?",
        "",
        f"**Run:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Corpus:** 57,638 passages · **Queries evaluated:** {len(query_ids)} of {n_total} test queries  ",
        f"**Retrieval depth:** {MATCH_COUNT} candidates per branch, top-{FINAL_K} output, RRF k={RRF_K}  ",
        f"**Reformulation LLM:** `{cache.get('metadata', {}).get('model', '?')}` · cache "
        f"`{os.path.relpath(cache_path, ROOT).replace(os.sep, '/')}` "
        f"({len(cache.get('parse_errors', {}))} unparseable replies fell back to the original query)",
        "",
        "## 1. Results",
        "",
        "| System | NDCG@10 | Recall@10 | MRR | Recall@20 | s/query |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for key, name in SYSTEMS.items():
        m = results[key]["mean"]
        lines.append(f"| {name} | {m['ndcg@10']:.4f} | {m['recall@10']:.4f} | {m['mrr']:.4f} | "
                     f"{m['recall@20']:.4f} | {results[key]['seconds_per_query']:.2f} |")

    lines += ["", "## 2. Significance (paired randomization test, 95% bootstrap CI, 10,000 resamples)", ""]
    for a, b in COMPARISONS:
        lines += [f"### {SYSTEMS[b]} vs {SYSTEMS[a]}", "",
                  "| Metric | A | B | Δ (B−A) | 95% CI | p | Significant |", "|:---|---:|---:|---:|:---:|---:|:---:|"]
        for m in METRICS:
            c = paired_comparison([results[a]["per_query"][q][m] for q in query_ids],
                                  [results[b]["per_query"][q][m] for q in query_ids])
            lines.append(f"| {m} | {c.mean_a:.4f} | {c.mean_b:.4f} | {c.mean_diff:+.4f} | "
                         f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}] | {c.p_value:.4f} | {'yes' if c.significant else 'no'} |")
        lines.append("")

    a, b = results["hybrid"]["per_query"], results["llm_pipeline"]["per_query"]
    better = sum(b[q]["ndcg@10"] > a[q]["ndcg@10"] + 1e-9 for q in query_ids)
    worse = sum(b[q]["ndcg@10"] < a[q]["ndcg@10"] - 1e-9 for q in query_ids)
    lines += [
        "## 3. Per-query effect of LLM reformulation (NDCG@10 vs hybrid)",
        "",
        f"Better: {better} · Worse: {worse} · Unchanged: {len(query_ids) - better - worse}",
        "",
        "## 4. Notes",
        "",
        "- All systems share the same BM25 index, dense index and retrieval depth; only the query side differs.",
        "- The pipeline is used unchanged from the university project, including its domain-tuned normalizer.",
        "- FiQA relevance labels are binary; NDCG@10 is the standard BEIR headline metric.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="BEIR FiQA evaluation of LLM query reformulation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("encode", help="Encode all passages with bge-small and cache the matrix.")
    r = sub.add_parser("reformulate", help="Generate and cache LLM reformulations (resumable).")
    r.add_argument("--limit", type=int, default=None, help="Only process this many new queries.")
    r.add_argument("--min-interval", type=float, default=13.0, help="Seconds between LLM calls.")
    e = sub.add_parser("evaluate", help="Evaluate all systems.")
    for p in (r, e):
        p.add_argument("--model", default=None, help="LLM model for reformulation (default: LLM_MODEL from .env).")
    e.add_argument("--allow-partial", action="store_true",
                   help="Evaluate only queries that already have cached reformulations.")
    args = parser.parse_args()
    with keep_awake():
        _run(args)


def _run(args: argparse.Namespace) -> None:

    chunks, queries, qrels = load_fiqa()
    if args.command == "encode":
        from embedding.encoder import EmbeddingEncoder

        print(f"Encoding {len(chunks)} passages (cached to {DENSE_CACHE}) ...", flush=True)
        t0 = time.perf_counter()
        index = DenseIndex(chunks, EmbeddingEncoder(), cache_path=DENSE_CACHE)
        print(f"Done: {index.matrix.shape} in {time.perf_counter() - t0:.0f}s", flush=True)
        return

    from reformulation.llm.config import LLMConfig

    config = LLMConfig.from_env()
    if args.model:
        config = dataclasses.replace(config, model=args.model)
    cache_path = cache_path_for(config.model)

    if args.command == "reformulate":
        from reformulation.llm.client import LLMClient
        from reformulation.reformulator import QueryReformulator

        reformulator = QueryReformulator(LLMClient(config))
        print(f"Reformulating {len(queries)} FiQA test queries with {config.model} "
              f"(one call every {args.min_interval:.0f}s) ...", flush=True)
        stats = generate_reformulations(
            queries, reformulator, cache_path=cache_path, min_interval=args.min_interval, limit=args.limit,
            metadata={"model": config.model, "provider": config.provider, "max_variants": MAX_VARIANTS,
                      "started": datetime.now(timezone.utc).isoformat()},
        )
        print(json.dumps(stats, indent=1))
        return

    cache = load_cache(cache_path)
    covered = [q for q in queries if q in cache["reformulations"]]
    if len(covered) < len(queries) and not args.allow_partial:
        sys.exit(f"Only {len(covered)}/{len(queries)} queries have reformulations. "
                 "Run `reformulate` until done, or pass --allow-partial.")
    from embedding.encoder import EmbeddingEncoder

    encoder = EmbeddingEncoder(query_cache_size=20000)
    systems = build_systems(chunks, queries, cache["reformulations"], encoder)
    results = evaluate(queries, qrels, systems, covered)
    suffix = f"_{os.path.basename(cache_path)[len('beir_fiqa_reformulations_'):-5]}"
    suffix += "" if len(covered) == len(queries) else "_partial"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"beir_fiqa_results{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"query_ids": covered, "results": results}, f)
    report = os.path.join(RESULTS_DIR, f"beir_fiqa_report{suffix}.md")
    write_report(results, covered, len(queries), cache, cache_path, report)
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
