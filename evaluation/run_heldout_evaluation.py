"""Held-out evaluation: what does each query-side layer add, on queries nobody tuned on?

Every system shares the same indexes and retrieval depth; only the query side
differs, so each comparison isolates one layer:

* ``hybrid``           – contextual vectors + BM25, raw query (no query processing)
* ``pipeline_no_llm``  – + normalization and corpus spelling correction
* ``pipeline_llm``     – + LLM reformulations, without spelling correction
* ``full``             – normalization + spelling + LLM reformulations (production)

Significance: paired randomization test per comparison and metric, with a
Holm correction across all tests in the report. Reformulations come from a
fixed cache (``evaluation/build_reformulation_cache.py``), so runs are
deterministic and need no LLM.

Usage::

    python evaluation/run_heldout_evaluation.py                          # held-out test set
    python evaluation/run_heldout_evaluation.py --golden evaluation/golden_queries.json \\
        --reformulations evaluation/cache/reformulations.json --tag tune # original 50 (tuning set)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Callable, Dict, List


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding.encoder import EmbeddingEncoder
from evaluation.run_phase5_evaluation import (
    CHUNKS_PATH, FINAL_K, MATCH_COUNT, MAX_VARIANTS, POOL_K, RRF_K,
    CachedReformulator, LocalVectorRetriever, evaluate,
)
from evaluation.significance import holm_adjust, paired_comparison
from reformulation.search_pipeline import SearchPipeline
from reformulation.semantic_similarity import SemanticSimilarity
from reformulation.spelling import CorpusSpellCorrector
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_GOLDEN = os.path.join(ROOT, "evaluation", "heldout_queries.json")
# The committed cache the published held-out report was produced from.
DEFAULT_REFORMULATIONS = os.path.join(
    ROOT, "evaluation", "cache", "heldout_queries_reformulations_gemini-3.5-flash-lite.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

SYSTEMS = {
    "bm25": "BM25 only",
    "vector": "Vector only (contextual passages)",
    "hybrid": "Hybrid, raw query (no query processing)",
    "pipeline_no_llm": "Pipeline: normalization + spelling, no LLM",
    "pipeline_llm": "Pipeline: normalization + LLM rewrites, no spelling",
    "full": "Full pipeline: normalization + spelling + LLM rewrites",
}
COMPARISONS = [
    ("hybrid", "pipeline_no_llm", "Normalization + spelling"),
    ("pipeline_no_llm", "full", "LLM rewrites (on top of spelling)"),
    ("hybrid", "full", "Whole query layer"),
]
METRICS = ("recall@10", "mrr", "ndcg@10")


class _NoReformulation:
    def generate_variants(self, query: str, max_variants: int = 5) -> List[str]:
        return []


def load_reformulations(path: str) -> Dict[str, List[str]]:
    """Accept both the plain ``{qid: [...]}`` cache and the ``{"reformulations": {...}}`` format."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("reformulations", data) if isinstance(data, dict) else {}


def build_systems(chunks, encoder, reformulator: CachedReformulator) -> Dict[str, Callable]:
    vec = LocalVectorRetriever(chunks, encoder, contextual=True)
    bm25 = BM25Retriever(chunks=chunks)
    sem = SemanticSimilarity(embed_fn=encoder.embed_query)
    speller = CorpusSpellCorrector.from_chunks_file(CHUNKS_PATH)

    def single(retriever):
        return lambda g, depth: [c.chunk_id for c in retriever.retrieve(g["query"], match_count=depth)]

    def hybrid():
        h = HybridRetriever(vector_retriever=vec, keyword_retriever=bm25)
        return lambda g, depth: [c.chunk_id for c in h.retrieve(
            g["query"], vector_match_count=max(MATCH_COUNT, depth), keyword_match_count=max(MATCH_COUNT, depth),
            k=RRF_K, final_k=depth)]

    def pipeline(ref, spell_corrector):
        p = SearchPipeline(reformulator=ref, semantic_similarity=sem, reformulation_cache_size=0,
                           hybrid_retriever=HybridRetriever(vector_retriever=vec, keyword_retriever=bm25),
                           spell_corrector=spell_corrector)
        return lambda g, depth: [c.chunk_id for c in p.search(
            g["query"], max_variants=MAX_VARIANTS, vector_match_count=max(MATCH_COUNT, depth),
            keyword_match_count=max(MATCH_COUNT, depth), k=RRF_K, final_k=depth, candidate_k=max(POOL_K, depth))]

    return {
        "bm25": single(bm25),
        "vector": single(vec),
        "hybrid": hybrid(),
        "pipeline_no_llm": pipeline(_NoReformulation(), speller),
        "pipeline_llm": pipeline(reformulator, None),
        "full": pipeline(reformulator, speller),
    }


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def write_report(results, golden, n_dropped, args, path) -> None:
    fmt = lambda x: f"{x:.4f}"
    lines = [
        f"# Held-out Evaluation — `{args.tag}`",
        "",
        f"**Run:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Queries:** {len(golden)} answerable (`{_rel(args.golden)}`"
        + (f"; {n_dropped} unanswerable excluded" if n_dropped else "") + ")  ",
        f"**Reformulations:** `{_rel(args.reformulations)}` · **Output depth:** top-{FINAL_K} · "
        f"**Pool depth:** top-{POOL_K} · **Per-branch candidates:** max({MATCH_COUNT}, depth) · **RRF k:** {RRF_K}",
        "",
        "## 1. Overall",
        "",
        "| System | R@5 | R@10 | R@20 | MRR | NDCG@10 | Pool R@50 |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name in SYSTEMS.items():
        m = results[key]["mean"]
        lines.append(f"| {name} | {fmt(m['recall@5'])} | {fmt(m['recall@10'])} | {fmt(m['recall@20'])} | "
                     f"{fmt(m['mrr'])} | {fmt(m['ndcg@10'])} | {fmt(m['pool_recall@50'])} |")

    tests = {
        (a, b, m): paired_comparison([q["metrics"][m] for q in results[a]["per_query"]],
                                     [q["metrics"][m] for q in results[b]["per_query"]])
        for a, b, _ in COMPARISONS for m in METRICS
    }
    holm = dict(zip(tests, holm_adjust([t.p_value for t in tests.values()])))
    lines += [
        "",
        "## 2. What each layer adds",
        "",
        "Paired randomization test (two-sided), 95% bootstrap CI, 10,000 resamples, seed 0. "
        f"Holm-adjusted p corrects for all {len(tests)} tests; \"Significant\" uses the adjusted p < 0.05.",
        "",
        "| Layer | Metric | Before | After | Δ | 95% CI | p | Holm p | Significant |",
        "|:---|:---|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for a, b, label in COMPARISONS:
        for m in METRICS:
            t, p_adj = tests[(a, b, m)], holm[(a, b, m)]
            lines.append(f"| {label} | {m} | {t.mean_a:.4f} | {t.mean_b:.4f} | {t.mean_diff:+.4f} | "
                         f"[{t.ci_low:+.4f}, {t.ci_high:+.4f}] | {t.p_value:.4f} | {p_adj:.4f} | "
                         f"{'yes' if p_adj < 0.05 else 'no'} |")

    base, final = results["hybrid"], results["full"]
    lines += ["", "## 3. Per category: hybrid (raw query) → full pipeline", "",
              "| Category | n | R@10 before | R@10 after | MRR before | MRR after | NDCG@10 before | NDCG@10 after |",
              "|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for cat, bm in base["categories"].items():
        fm = final["categories"][cat]
        n = sum(1 for g in golden if g["category"] == cat)
        lines.append(f"| `{cat}` | {n} | {fmt(bm['recall@10'])} | {fmt(fm['recall@10'])} | {fmt(bm['mrr'])} | "
                     f"{fmt(fm['mrr'])} | {fmt(bm['ndcg@10'])} | {fmt(fm['ndcg@10'])} |")

    no_llm = {q["query_id"]: q["metrics"]["ndcg@10"] for q in results["pipeline_no_llm"]["per_query"]}
    deltas = [(q["metrics"]["ndcg@10"] - no_llm[q["query_id"]], q) for q in final["per_query"]]
    better = sum(d > 1e-9 for d, _ in deltas)
    worse = sorted(((d, q) for d, q in deltas if d < -1e-9), key=lambda x: x[0])
    lines += ["", "## 4. Per-query effect of the LLM rewrites (NDCG@10, full vs no-LLM pipeline)", "",
              f"Better: {better} · Worse: {len(worse)} · Unchanged: {len(deltas) - better - len(worse)}", ""]
    if worse:
        lines += ["| Query | Category | Text | Δ NDCG@10 |", "|:---|:---|:---|---:|"]
        lines += [f"| {q['query_id']} | `{q['category']}` | {q['query']} | {d:+.3f} |" for d, q in worse[:15]]

    lines += [
        "",
        "## 5. Notes",
        "",
        "- One query moves a mean by 1/n; with ~100 queries treat differences below ~0.02 as noise.",
        "- Only the query side differs between systems; indexes, depths and fusion are identical.",
        "- MRR is computed over the top-20 output (MRR@20).",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out evaluation of the query-side layers")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--reformulations", default=None,
                        help="Reformulation cache (default: the committed held-out cache; "
                             "for other query files, the cache for LLM_MODEL).")
    parser.add_argument("--tag", default=None,
                        help="Name used in output files (default: 'heldout', or the golden file stem).")
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    args.golden = os.path.abspath(args.golden)
    if args.tag is None:
        args.tag = "heldout" if args.golden == DEFAULT_GOLDEN else os.path.splitext(os.path.basename(args.golden))[0]
    if args.reformulations is None and args.golden == DEFAULT_GOLDEN:
        args.reformulations = DEFAULT_REFORMULATIONS
    if args.reformulations is None:
        from dotenv import load_dotenv
        from evaluation.build_reformulation_cache import cache_path_for

        load_dotenv(os.path.join(ROOT, ".env"))
        args.reformulations = cache_path_for(args.golden, os.getenv("LLM_MODEL", "unknown"))
    args.reformulations = os.path.abspath(args.reformulations)

    with open(args.golden, encoding="utf-8") as f:
        all_queries = json.load(f)
    golden = [g for g in all_queries if g.get("relevant_chunks")]
    reformulations = load_reformulations(args.reformulations)
    missing = [g["query_id"] for g in golden if g["query_id"] not in reformulations]
    if missing:
        sys.exit(f"{len(missing)} queries have no cached reformulations (e.g. {missing[:5]}); "
                 "run evaluation/build_reformulation_cache.py first.")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    reformulator = CachedReformulator(reformulations)
    systems = build_systems(chunks, EmbeddingEncoder(), reformulator)
    results: Dict[str, dict] = {}
    for key, fn in systems.items():
        print(f"Evaluating: {SYSTEMS[key]} ...", flush=True)
        results[key] = evaluate(golden, fn, reformulator)
        m = results[key]["mean"]
        print(f"    R@10={m['recall@10']:.3f}  MRR={m['mrr']:.3f}  NDCG@10={m['ndcg@10']:.3f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump({"systems": SYSTEMS, "results": results}, f, indent=1)
    report = os.path.join(args.output_dir, f"{args.tag}_report.md")
    write_report(results, golden, len(all_queries) - len(golden), args, report)
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
