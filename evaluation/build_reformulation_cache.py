"""Build a fixed LLM reformulation cache for a golden-query file.

Evaluations read reformulations from a cache instead of calling the LLM, so a
run is deterministic and repeatable. This script fills that cache: paced,
resumable (re-run to continue after a quota stop) and one file per model.

Usage::

    python evaluation/build_reformulation_cache.py --queries evaluation/heldout_queries.json
    python evaluation/build_reformulation_cache.py --queries evaluation/heldout_queries.json --model gemini-3.5-flash-lite
    python evaluation/build_reformulation_cache.py --queries evaluation/heldout_queries.json --glossary  # organisation-vocabulary prompt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.run_beir_evaluation import MAX_VARIANTS, generate_reformulations, keep_awake

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT, "evaluation", "cache")


def cache_path_for(queries_path: str, model: str, glossary: bool = False) -> str:
    """``evaluation/cache/<queries file stem>_reformulations_<model>[_glossary].json``."""
    stem = os.path.splitext(os.path.basename(queries_path))[0]
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model.split("/")[-1])
    return os.path.join(CACHE_DIR, f"{stem}_reformulations_{safe}{'_glossary' if glossary else ''}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill an LLM reformulation cache for golden queries")
    parser.add_argument("--queries", required=True, help="Golden-query JSON file (list of {query_id, query, ...}).")
    parser.add_argument("--model", default=None, help="LLM model (default: LLM_MODEL from .env).")
    parser.add_argument("--min-interval", type=float, default=13.0, help="Seconds between LLM calls.")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many new queries.")
    parser.add_argument("--glossary", action="store_true",
                        help="Use the organisation-glossary prompt (separate cache file).")
    args = parser.parse_args()

    from reformulation.llm.client import LLMClient
    from reformulation.llm.config import LLMConfig
    from reformulation.glossary import load_glossary
    from reformulation.reformulator import QueryReformulator

    with open(args.queries, encoding="utf-8") as f:
        queries = {q["query_id"]: q["query"] for q in json.load(f)}
    config = LLMConfig.from_env()
    if args.model:
        config = dataclasses.replace(config, model=args.model)
    cache_path = cache_path_for(args.queries, config.model, glossary=args.glossary)
    reformulator = QueryReformulator(LLMClient(config), glossary=load_glossary() if args.glossary else None)

    print(f"Reformulating {len(queries)} queries with {config.model} -> {cache_path}", flush=True)
    with keep_awake():
        stats = generate_reformulations(
            queries, reformulator, cache_path=cache_path,
            min_interval=args.min_interval, limit=args.limit,
            metadata={"model": config.model, "provider": config.provider, "max_variants": MAX_VARIANTS,
                      "prompt": reformulator.prompt_version,
                      "queries_file": os.path.relpath(args.queries, ROOT).replace(os.sep, "/"),
                      "started": datetime.now(timezone.utc).isoformat()},
        )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
