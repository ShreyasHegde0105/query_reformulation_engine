"""Benchmark configurations A-D under identical conditions.

    A  original query  + hybrid retrieval, no reranker
    B  reformulated    + hybrid retrieval, no reranker
    C  original query  + hybrid retrieval + production reranker
    D  reformulated    + hybrid retrieval + production reranker

All four use the same corpus snapshot, the same labels, the same per-branch depth and
candidate pool (``--candidate-limit``), and the same output depth (``--final-k``). The only
differences are rewrites on/off and the reranker on/off. Intent routing is disabled in all
four (it was tuned on the contaminated validation set). Rewrites come from a fixed cache, so
runs are deterministic; the live LLM call is therefore NOT included in the latency figures.

Metrics come ONLY from the split's labels. The optional LLM judge (``--judge-sample N``) is
run afterwards on the pooled top-10 passages, blind to configuration; its grades are stored
separately and reported only as agreement with the existing labels. They never change any
metric and never declare an improvement.

    python evaluation/run_ad_benchmark.py --split validation_contaminated --candidate-limit 50
    python evaluation/run_ad_benchmark.py --split development --reranker-model BAAI/bge-reranker-base
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_LOCAL_FILES_ONLY", "1")

from evaluation.datasets.splits import BlindTestNotFrozenError, load_split, sha256_file  # noqa: E402
from evaluation.metrics import calculate_ndcg_at_k  # noqa: E402
from evaluation.significance import holm_adjust, paired_comparison  # noqa: E402

CHUNKS_PATH = os.path.join(ROOT, "Data", "chunks", "chunks.jsonl")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
REWRITE_CACHES = {
    "development": os.path.join(ROOT, "evaluation", "cache", "reformulations.json"),
    "validation_contaminated": os.path.join(ROOT, "evaluation", "cache",
                                            "heldout_queries_reformulations_gemini-3.5-flash-lite.json"),
    "blind_test": os.path.join(ROOT, "evaluation", "cache", "blind_test_v1_reformulations_gemini-3.5-flash-lite.json"),
}
CONFIGS = {
    "A": "original + hybrid, no reranker",
    "B": "reformulated + hybrid, no reranker",
    "C": "original + hybrid + reranker",
    "D": "reformulated + hybrid + reranker",
}
COMPARISONS = [("A", "B", "reformulation, no reranker"), ("A", "C", "reranker on original"),
               ("A", "D", "reformulation + reranker"), ("C", "D", "reformulation given reranker"),
               ("B", "D", "reranker given reformulation")]
TEST_METRICS = ("P@10", "R@10", "MRR@10", "NDCG@10")
RRF_K = 60
MAX_VARIANTS = 5


# ----------------------------------------------------------------------------- metrics
def query_metrics(ranked: List[str], g: Dict[str, Any], chunk_doc: Dict[str, str]) -> Dict[str, float]:
    rel = set(g["relevant_chunks"])
    graded = g.get("graded_relevance") or {c: 1 for c in rel}
    m: Dict[str, float] = {}
    for k in (5, 10):
        hits = len(rel & set(ranked[:k]))
        m[f"P@{k}"] = hits / k
        m[f"R@{k}"] = hits / len(rel)
    m["Hit@10"] = 1.0 if rel & set(ranked[:10]) else 0.0
    m["MRR@10"] = next((1.0 / i for i, c in enumerate(ranked[:10], 1) if c in rel), 0.0)
    m["NDCG@10"] = calculate_ndcg_at_k(ranked, graded, 10)
    docs = [chunk_doc.get(c) for c in ranked[:10]]
    m["UniqueDocs@10"] = float(len(set(docs)))
    m["DupDocRate@10"] = (len(docs) - len(set(docs))) / len(docs) if docs else 0.0
    return m


def git_state() -> Dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


# ----------------------------------------------------------------------------- systems
def build(chunks, reformulations, reranker_model: str, candidate_limit: int, judge_unused=None):
    from embedding.encoder import EmbeddingEncoder
    from evaluation.run_phase5_evaluation import CachedReformulator, LocalVectorRetriever
    from reformulation.intent_router import IntentRouteResult, QueryIntent, RoutingAction
    from reformulation.production_reranker import ProductionReranker, RerankerConfig
    from reformulation.search_pipeline import SearchPipeline
    from reformulation.semantic_similarity import SemanticSimilarity
    from reformulation.spelling import CorpusSpellCorrector
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.hybrid_retriever import HybridRetriever

    class NoRouting:
        def route(self, query, normalized_query=None):
            return IntentRouteResult(query=query, normalized_query=normalized_query or query,
                                     intent=QueryIntent.NATURAL_LANGUAGE, action=RoutingAction.REFORMULATE,
                                     confidence=1.0, reason="routing disabled for A-D benchmark")

    class NoRewrites:
        def generate_variants(self, query, max_variants=5):
            return []

    encoder = EmbeddingEncoder()
    base_vec = LocalVectorRetriever(chunks, encoder, contextual=True)
    bm25 = BM25Retriever(chunks=chunks)
    sem = SemanticSimilarity(embed_fn=encoder.embed_query)
    speller = CorpusSpellCorrector.from_chunks_file(CHUNKS_PATH)
    cached = CachedReformulator(reformulations)
    reranker = ProductionReranker(RerankerConfig(enabled=True, model_name=reranker_model,
                                                 candidate_limit=candidate_limit))

    def make(rewrites: bool, rerank: bool):
        vec = LocalVectorRetriever.__new__(LocalVectorRetriever)   # shares the passage matrix ...
        vec.chunks, vec.encoder, vec.matrix = base_vec.chunks, encoder, base_vec.matrix
        vec._query_cache = {}                                       # ... but not query embeddings (fair latency)
        return SearchPipeline(reformulator=cached if rewrites else NoRewrites(), semantic_similarity=sem,
                              hybrid_retriever=HybridRetriever(vector_retriever=vec, keyword_retriever=bm25),
                              spell_corrector=speller, intent_router=NoRouting(), reformulation_cache_size=0,
                              reranker=reranker if rerank else None, max_workers=1)

    pipelines = {"A": make(False, False), "B": make(True, False), "C": make(False, True), "D": make(True, True)}
    return pipelines, cached, reranker


# ----------------------------------------------------------------------------- judge (optional)
def run_judge(golden, outputs, chunks_by_id, sample: int, seed: int) -> Dict[str, Any]:
    from dotenv import load_dotenv

    from evaluation.judge import JudgeConfig, LLMRelevanceJudge

    load_dotenv(os.path.join(ROOT, ".env"))
    cfg = JudgeConfig.from_env()
    judge = LLMRelevanceJudge(cfg.build_client(), cfg)
    rng = random.Random(seed)
    picked = rng.sample(golden, min(sample, len(golden)))
    results, pairs = [], []
    for g in picked:
        pool = list(dict.fromkeys(c for name in CONFIGS for c in outputs[name][g["query_id"]][:10]))
        res = judge.judge(g["query_id"], g["query"], [chunks_by_id[c] for c in pool])
        results.append(res.to_dict())
        if res.status != "ok":
            continue
        labelled = g.get("graded_relevance") or {}
        for cid, grade in res.grades().items():
            pairs.append({"query_id": g["query_id"], "chunk_id": cid, "judge": grade,
                          "label": int(labelled.get(cid, 0)), "label_was_explicit": cid in labelled})
        time.sleep(float(os.getenv("JUDGE_PAUSE_SECONDS", "2")))
    return {"judge_model": judge.model, "judge_provider": judge.provider, "sample_query_ids": [g["query_id"] for g in picked],
            "results": results, "pairs": pairs, "agreement": agreement(pairs)}


def _kappa(a: List[int], b: List[int], weights: str) -> Optional[float]:
    cats = sorted(set(a) | set(b) | {0, 1, 2, 3})
    n = len(a)
    if n == 0:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= n
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.array([[((i - j) ** 2 if weights == "quadratic" else float(i != j)) for j in range(len(cats))]
                  for i in range(len(cats))])
    denom = (w * exp).sum()
    return None if denom == 0 else float(1 - (w * obs).sum() / denom)


def agreement(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not pairs:
        return {"status": "NOT EVALUATED (no accepted judgments)"}
    j = [p["judge"] for p in pairs]
    l = [p["label"] for p in pairs]
    jb, lb = [int(x >= 2) for x in j], [int(x >= 2) for x in l]
    conf = {f"label{a}_judge{b}": sum(1 for p in pairs if p["label"] == a and p["judge"] == b)
            for a in range(4) for b in range(4)}
    disagreements = sorted(pairs, key=lambda p: -abs(p["judge"] - p["label"]))[:15]
    return {
        "n_pairs": len(pairs),
        "reference": "existing split labels (provenance UNRECORDED; unlabelled chunks treated as grade 0). "
                     "This is NOT judge-human agreement.",
        "exact_agreement": float(np.mean([a == b for a, b in zip(j, l)])),
        "quadratic_weighted_kappa": _kappa(l, j, "quadratic"),
        "binary_relevant_agreement": float(np.mean([a == b for a, b in zip(jb, lb)])),
        "binary_relevant_kappa": _kappa(lb, jb, "unweighted"),
        "confusion": conf,
        "largest_disagreements": disagreements,
    }


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["development", "validation_contaminated", "blind_test"], required=True)
    ap.add_argument("--candidate-limit", type=int, default=50, choices=[20, 50, 100])
    ap.add_argument("--final-k", type=int, default=10)
    ap.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--judge-sample", type=int, default=0, help="queries to grade with the LLM judge (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    try:
        queries, split_info = load_split(args.split)
    except BlindTestNotFrozenError as exc:
        print(f"{args.split}: NOT EVALUATED ({exc})")
        return 2
    golden = [q for q in queries if q.get("relevant_chunks")]
    unanswerable = [q for q in queries if not q.get("relevant_chunks")]

    cache_path = REWRITE_CACHES[args.split]
    rdata = json.load(open(cache_path, encoding="utf-8"))
    reformulations = rdata.get("reformulations", rdata)
    missing = [q["query_id"] for q in queries if q["query_id"] not in reformulations]
    if missing:
        print(f"{len(missing)} queries lack cached rewrites (e.g. {missing[:3]}); run build_reformulation_cache.py")
        return 2

    chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8") if l.strip()]
    chunk_doc = {c["chunk_id"]: c["document_id"] for c in chunks}
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    pipelines, cached, reranker = build(chunks, reformulations, args.reranker_model, args.candidate_limit)

    match = max(20, args.candidate_limit)       # identical branch depth for all four configurations
    def run(name, q):
        cached.query_id = q["query_id"]
        return pipelines[name].search_detailed(q["query"], max_variants=MAX_VARIANTS, vector_match_count=match,
                                               keyword_match_count=match, k=RRF_K, final_k=args.final_k,
                                               candidate_k=args.candidate_limit)

    # Warm-up (model loading) so latency measures steady state.
    for name in CONFIGS:
        run(name, queries[0])

    per: Dict[str, List[Dict[str, Any]]] = {n: [] for n in CONFIGS}
    outputs: Dict[str, Dict[str, List[str]]] = {n: {} for n in CONFIGS}
    for name in CONFIGS:
        for q in queries:
            res = run(name, q)
            ranked = [c.chunk_id for c in res.results]
            outputs[name][q["query_id"]] = ranked
            rec = {"query_id": q["query_id"], "category": q.get("category"), "ranked": ranked,
                   "latency_ms": {"retrieval": res.timings_ms.get("retrieval", 0.0),
                                  "reranking": res.timings_ms.get("reranking", 0.0),
                                  "end_to_end_excl_llm": res.timings_ms.get("total", 0.0)},
                   "reranking": res.reranking, "queries_used": res.queries_used}
            if q.get("relevant_chunks"):
                rec["metrics"] = query_metrics(ranked, q, chunk_doc)
            per[name].append(rec)
        print(f"  {name}: done", flush=True)

    def mean_metrics(name):
        rows = [r["metrics"] for r in per[name] if "metrics" in r]
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}

    def latency(name):
        out = {}
        for key in ("retrieval", "reranking", "end_to_end_excl_llm"):
            xs = [r["latency_ms"][key] for r in per[name]]
            out[key] = {"mean": float(np.mean(xs)), "p95": float(np.percentile(xs, 95))}
        return out

    summary = {n: {"metrics": mean_metrics(n), "latency_ms": latency(n)} for n in CONFIGS}
    tests = {}
    for a, b, _ in COMPARISONS:
        for m in TEST_METRICS:
            ra = [r["metrics"][m] for r in per[a] if "metrics" in r]
            rb = [r["metrics"][m] for r in per[b] if "metrics" in r]
            tests[(a, b, m)] = paired_comparison(ra, rb) if ra else None
    adj = dict(zip([k for k, v in tests.items() if v], holm_adjust([v.p_value for v in tests.values() if v])))
    comparisons = []
    for (a, b, m), t in tests.items():
        if t is None:
            continue
        base = summary[a]["metrics"][m]
        comparisons.append({"baseline": a, "system": b, "metric": m, "baseline_value": base,
                            "system_value": summary[b]["metrics"][m], "abs_change": t.mean_diff,
                            "rel_change_pct": (t.mean_diff / base * 100) if base else None,
                            "ci95": [t.ci_low, t.ci_high], "p": t.p_value, "holm_p": adj[(a, b, m)]})

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "git": git_state(), "python": platform.python_version(),
        "split": args.split, "split_status": split_info.get("status"), "label_provenance": split_info.get("label_provenance"),
        "queries_file": split_info["queries_file"], "queries_sha256": split_info["queries_sha256"],
        "n_queries": len(queries), "n_labelled": len(golden), "n_unlabelled_or_unanswerable": len(unanswerable),
        "corpus_file": "Data/chunks/chunks.jsonl", "corpus_sha256": sha256_file(CHUNKS_PATH), "n_chunks": len(chunks),
        "rewrite_cache": os.path.relpath(cache_path, ROOT), "rewrite_cache_sha256": sha256_file(cache_path),
        "rewrite_model": (rdata.get("metadata") or {}).get("model", "unrecorded"),
        "rewrite_prompt_version": ((rdata.get("metadata") or {}).get("prompt_version") or (rdata.get("metadata") or {}).get("prompt")
                                   or "generic-v1 (assumed; not recorded in cache)"),
        "embedding_model": "BAAI/bge-small-en-v1.5 (contextual passages, exact cosine)",
        "reranker": dataclasses.asdict(reranker.config),
        "config": {"candidate_limit": args.candidate_limit, "final_k": args.final_k, "branch_depth": match,
                   "rrf_k": RRF_K, "max_variants": MAX_VARIANTS, "intent_routing": "disabled",
                   "spell_corrector": "corpus (all configs)"},
        "latency_note": "In-process exact vector search + BM25 on CPU; the live LLM rewrite call is NOT included "
                        "(cached rewrites). Live LLM latency was measured separately (audit 2026-11-13: ~1.1 s/call, n=8).",
    }
    out: Dict[str, Any] = {"meta": meta, "configs": CONFIGS, "summary": summary, "comparisons": comparisons,
                           "per_query": per}
    if args.judge_sample:
        out["judge"] = run_judge(golden, outputs, chunks_by_id, args.judge_sample, args.seed)

    tag = args.tag or f"{args.split}_c{args.candidate_limit}_{args.reranker_model.split('/')[-1]}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jpath = os.path.join(RESULTS_DIR, f"ad_benchmark_{tag}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    write_report(out, os.path.join(RESULTS_DIR, f"ad_benchmark_{tag}.md"))
    print("wrote", jpath)
    return 0


def write_report(out: Dict[str, Any], path: str) -> None:
    meta, s = out["meta"], out["summary"]
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    L = [f"# A-D benchmark: `{meta['split']}`", "",
         f"- **Split status:** {meta['split_status']}. **Label provenance:** {meta['label_provenance']}",
         f"- **Queries:** {meta['n_queries']} ({meta['n_labelled']} with labelled relevant chunks; "
         f"{meta['n_unlabelled_or_unanswerable']} without, excluded from metrics)",
         f"- **Corpus:** `{meta['corpus_file']}` sha256 `{meta['corpus_sha256'][:16]}…` ({meta['n_chunks']} chunks) · "
         f"**Git:** `{(meta['git']['commit'] or '?')[:10]}`{' (dirty)' if meta['git']['dirty'] else ''}",
         f"- **Rewrites:** `{meta['rewrite_cache']}` ({meta['rewrite_model']}, {meta['rewrite_prompt_version']})",
         f"- **Reranker:** `{meta['reranker']['model_name']}` · **candidate limit:** {meta['config']['candidate_limit']} · "
         f"**final k:** {meta['config']['final_k']} · branch depth {meta['config']['branch_depth']} · RRF k {meta['config']['rrf_k']} · routing off",
         f"- {meta['latency_note']}", "",
         "| Config | P@5 | P@10 | R@5 | R@10 | MRR@10 | NDCG@10 | Hit@10 | Unique docs@10 | Dup-doc rate | Retrieval ms (mean/p95) | Rerank ms (mean/p95) | End-to-end ms excl. LLM (mean/p95) |",
         "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, desc in out["configs"].items():
        m, lat = s[n]["metrics"], s[n]["latency_ms"]
        L.append(f"| **{n}** {desc} | {f3(m.get('P@5'))} | {f3(m.get('P@10'))} | {f3(m.get('R@5'))} | {f3(m.get('R@10'))} | "
                 f"{f3(m.get('MRR@10'))} | {f3(m.get('NDCG@10'))} | {f3(m.get('Hit@10'))} | {m.get('UniqueDocs@10', 0):.2f} | "
                 f"{f3(m.get('DupDocRate@10'))} | {lat['retrieval']['mean']:.0f} / {lat['retrieval']['p95']:.0f} | "
                 f"{lat['reranking']['mean']:.0f} / {lat['reranking']['p95']:.0f} | "
                 f"{lat['end_to_end_excl_llm']['mean']:.0f} / {lat['end_to_end_excl_llm']['p95']:.0f} |")
    L += ["", "## Paired comparisons (randomization test, 95% bootstrap CI, Holm across all rows)", "",
          "| Baseline → system | Metric | Baseline | System | Abs. Δ | Rel. Δ | 95% CI | Holm p |",
          "|:---|:---|---:|---:|---:|---:|:---:|---:|"]
    for c in out["comparisons"]:
        rel = "n/a (baseline 0)" if c["rel_change_pct"] is None else f"{c['rel_change_pct']:+.1f}%"
        L.append(f"| {c['baseline']} → {c['system']} | {c['metric']} | {c['baseline_value']:.3f} | {c['system_value']:.3f} | "
                 f"{c['abs_change']:+.3f} | {rel} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['holm_p']:.3f} |")
    if "judge" in out:
        a = out["judge"]["agreement"]
        L += ["", "## LLM judge (diagnostic only; does not affect any metric above)", "",
              f"- Judge: `{out['judge']['judge_provider']}/{out['judge']['judge_model']}`, "
              f"{len(out['judge']['sample_query_ids'])} sampled queries, pooled top-10 of A-D, blind to configuration.",
              f"- Accepted/rejected: {sum(r['status'] == 'ok' for r in out['judge']['results'])} / "
              f"{sum(r['status'] != 'ok' for r in out['judge']['results'])}"]
        if "n_pairs" in a:
            L += [f"- Reference: {a['reference']}",
                  f"- Pairs: {a['n_pairs']} · exact agreement {a['exact_agreement']:.3f} · quadratic-weighted κ "
                  f"{f3(a['quadratic_weighted_kappa'])} · binary (grade ≥ 2) agreement {a['binary_relevant_agreement']:.3f}, "
                  f"κ {f3(a['binary_relevant_kappa'])}"]
        else:
            L.append(f"- Agreement: {a['status']}")
        L.append("- Judge-human agreement: NOT EVALUATED (no labels with recorded human provenance).")
    L += ["", "No improvement is claimed from this report alone: see the split status above."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
