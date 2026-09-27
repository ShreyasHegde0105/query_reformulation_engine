from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.evaluator import RetrievalEvaluator
from evaluation.schema import EvaluationConfig, EvaluationSummary, GoldenQuery
from reformulation.llm.client import LLMClient
from reformulation.llm.config import LLMConfig
from reformulation.reformulator import QueryReformulator
from reformulation.search_pipeline import SearchPipeline
from reformulation.semantic_similarity import SemanticSimilarity
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluation.runner")


def load_golden_dataset(path: Optional[str] = None) -> List[GoldenQuery]:
    """Load and parse the golden evaluation dataset JSON."""
    default_path = os.path.join(os.path.dirname(__file__), "golden_queries.json")
    dataset_path = path or default_path
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Golden dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    queries = [GoldenQuery.from_dict(item) for item in raw_data]
    logger.info("Loaded %d golden queries from %s", len(queries), dataset_path)
    return queries


def generate_markdown_report(
    summaries: Dict[str, EvaluationSummary],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a clean, human-readable Markdown evaluation report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta = metadata or {}

    lines: List[str] = [
        "# Retrieval Quality Evaluation Report",
        "",
        f"**Run Timestamp:** {meta.get('timestamp', datetime.now(timezone.utc).isoformat())}",
        f"**Corpus Size:** {meta.get('corpus_chunks', 696)} document chunks",
        f"**Embedding Model:** {meta.get('embedding_model', 'BAAI/bge-small-en-v1.5 (384-d)')}",
        f"**RRF Damping (k):** {meta.get('rrf_k', 60)}",
        f"**Total Evaluation Queries:** {meta.get('total_queries', 50)}",
        "",
        "## 1. Overall Variant Benchmark",
        "",
        "| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 | NDCG@10 |",
        "|:---|---:|---:|---:|---:|---:|",
    ]

    variant_display_names = {
        "baseline_vector": "Original + Vector",
        "baseline_keyword": "Original + BM25",
        "baseline_hybrid": "Original + Hybrid",
        "reformulation_hybrid": "Reformulation + Hybrid",
        "full_pipeline": "Full Pipeline (Ours)",
    }

    for variant_key, summary in summaries.items():
        name = variant_display_names.get(variant_key, variant_key)
        m = summary.mean_metrics
        rec5 = f"{m.get('recall@5', 0.0):.4f}"
        rec10 = f"{m.get('recall@10', 0.0):.4f}"
        mrr = f"{m.get('mrr', 0.0):.4f}"
        ndcg5 = f"{m.get('ndcg@5', 0.0):.4f}"
        ndcg10 = f"{m.get('ndcg@10', 0.0):.4f}"
        lines.append(f"| {name} | {rec5} | {rec10} | {mrr} | {ndcg5} | {ndcg10} |")

    lines.extend([
        "",
        "## 2. Category-Specific Breakdown",
        "",
    ])

    all_categories = sorted(
        list(
            set(
                cat
                for s in summaries.values()
                for cat in s.category_metrics.keys()
            )
        )
    )

    for cat in all_categories:
        lines.extend([
            f"### Category: `{cat}`",
            "",
            "| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |",
            "|:---|---:|---:|---:|---:|",
        ])
        for variant_key, summary in summaries.items():
            name = variant_display_names.get(variant_key, variant_key)
            cm = summary.category_metrics.get(cat, {})
            r5 = f"{cm.get('recall@5', 0.0):.4f}"
            r10 = f"{cm.get('recall@10', 0.0):.4f}"
            mrr_val = f"{cm.get('mrr', 0.0):.4f}"
            n5 = f"{cm.get('ndcg@5', 0.0):.4f}"
            lines.append(f"| {name} | {r5} | {r10} | {mrr_val} | {n5} |")
        lines.append("")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Saved Markdown evaluation report to: %s", output_path)
    return content


def save_json_results(
    summaries: Dict[str, EvaluationSummary],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save raw machine-readable JSON evaluation results."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "summaries": {k: v.to_dict() for k, v in summaries.items()},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved JSON evaluation results to: %s", output_path)


def initialize_real_evaluator(config: Optional[EvaluationConfig] = None) -> RetrievalEvaluator:
    """Initialize evaluator wired to real project components."""
    vec = VectorRetriever()
    kw = BM25Retriever()
    hyb = HybridRetriever(vector_retriever=vec, keyword_retriever=kw)
    
    try:
        llm_cfg = LLMConfig.from_env()
        llm = LLMClient(config=llm_cfg)
        ref = QueryReformulator(llm_client=llm)
    except Exception as exc:
        logger.warning("Could not initialize real LLMClient: %s. Using mock/cached reformulations.", exc)
        from unittest.mock import MagicMock
        ref = MagicMock(spec=QueryReformulator)
        ref.generate_variants.return_value = []

    sem = SemanticSimilarity()
    pipe = SearchPipeline(
        reformulator=ref,
        semantic_similarity=sem,
        hybrid_retriever=hyb,
    )

    evaluator = RetrievalEvaluator(
        vector_retriever=vec,
        keyword_retriever=kw,
        hybrid_retriever=hyb,
        search_pipeline=pipe,
        config=config,
    )
    return evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Engine Retrieval Quality Evaluation Runner")
    parser.add_argument("--dataset", type=str, default=None, help="Path to golden queries JSON dataset.")
    parser.add_argument("--sample-size", type=int, default=None, help="Evaluate on a small sample subset of queries.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline_vector", "baseline_keyword", "baseline_hybrid", "reformulation_hybrid", "full_pipeline"],
        help="List of variants to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation artifacts (default: evaluation/results, or evaluation/results/sample with --sample-size).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable reformulation caching.")
    
    args = parser.parse_args()
    if args.output_dir is None:
        # Keep partial sample runs from overwriting the full benchmark results.
        args.output_dir = "evaluation/results/sample" if args.sample_size else "evaluation/results"

    # 1. Load dataset
    dataset = load_golden_dataset(args.dataset)
    if args.sample_size:
        dataset = dataset[:args.sample_size]
        logger.info("Sample mode: evaluating on first %d queries", len(dataset))

    # 2. Config
    config = EvaluationConfig(
        use_cache=not args.no_cache,
    )

    # 3. Initialize Evaluator
    evaluator = initialize_real_evaluator(config)

    # 4. Metadata
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(dataset),
        "corpus_chunks": 696,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "top_k_values": config.top_k_values,
        "rrf_k": config.rrf_k,
        "variants_evaluated": args.variants,
        "sample_mode": args.sample_size is not None,
    }

    # 5. Run evaluation
    summaries = evaluator.evaluate_all_variants(dataset, args.variants, metadata=metadata)

    # 6. Save outputs
    json_path = os.path.join(args.output_dir, "evaluation_results.json")
    md_path = os.path.join(args.output_dir, "evaluation_report.md")

    save_json_results(summaries, json_path, metadata=metadata)
    generate_markdown_report(summaries, md_path, metadata=metadata)

    print("\n" + "=" * 60)
    print("EVALUATION RUN COMPLETE")
    print("=" * 60)
    print(f"Total Queries Evaluated: {len(dataset)}")
    print(f"Results JSON: {json_path}")
    print(f"Report MD:    {md_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
