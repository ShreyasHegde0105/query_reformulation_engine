"""Build a laptop-sized MS MARCO passage benchmark in BEIR format.

The full collection (8.8M passages) does not fit in memory for in-process BM25
and exact dense search, so this builds the standard re-ranking-pool subset:

1. Sample ``--n-queries`` dev queries (seeded) that have relevance labels.
2. Stream the full collection twice to score Okapi BM25 for those queries only
   (same tokenizer, stemming and k1/b as ``retrieval.bm25_retriever``) and keep
   the top ``--depth`` passages per query. Memory stays small: only document
   frequencies of query terms and per-query heaps are held.
3. Write ``corpus.jsonl`` (the union of all pools + every judged-relevant passage
   of the sampled queries), ``queries.jsonl`` and ``qrels/dev.tsv`` in BEIR format.

The pool contains realistic near-miss passages (lexically similar to the query),
not random distractors, so the task stays hard. Its bias: the pool comes from the
original query's BM25 ranking; systems still see every judged-relevant passage.

Usage::

    # unzip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip into Data/beir/
    python evaluation/prepare_msmarco_sample.py            # -> Data/beir/msmarco-sample/
    python evaluation/run_beir_evaluation.py encode --dataset msmarco-sample
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Iterator, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.bm25_retriever import _STOP_WORDS, _TOKEN_RE, _stem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_DIR = os.path.join(ROOT, "Data", "beir", "msmarco")
OUTPUT_DIR = os.path.join(ROOT, "Data", "beir", "msmarco-sample")
K1, B = 1.2, 0.75  # BM25Retriever defaults

_stem_cached = lru_cache(maxsize=2_000_000)(_stem)


def tokenize(text: str) -> List[str]:
    """Same tokens as ``retrieval.bm25_retriever.tokenize``, with a stemming cache for speed."""
    return [_stem_cached(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def sample_query_ids(qrels: Dict[str, Dict[str, int]], n: int, seed: int) -> List[str]:
    ids = sorted(qrels)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def iter_corpus(path: str) -> Iterator[Tuple[str, dict, List[str]]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield d["_id"], d, tokenize(f"{d.get('title') or ''} {d['text']}")


def bm25_pools(corpus_path: str, queries: Dict[str, str], depth: int,
               log_every: int = 500_000) -> Dict[str, List[Tuple[float, str]]]:
    """Top-``depth`` BM25 passages per query over the whole collection (two streaming passes)."""
    qids = list(queries)
    term_queries: Dict[str, List[int]] = defaultdict(list)
    for qi, qid in enumerate(qids):
        for term in set(tokenize(queries[qid])):
            term_queries[term].append(qi)

    # Pass 1: collection size, average length, document frequency of query terms.
    n_docs, total_len, df = 0, 0, Counter()
    t0 = time.perf_counter()
    for i, (_, _, toks) in enumerate(iter_corpus(corpus_path), start=1):
        n_docs += 1
        total_len += len(toks)
        df.update(t for t in set(toks) if t in term_queries)
        if i % log_every == 0:
            print(f"  pass 1: {i:,} passages ({time.perf_counter() - t0:.0f}s)", flush=True)
    avg_len = total_len / max(n_docs, 1)
    idf = {t: math.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5)) for t in term_queries}

    # Pass 2: score only (passage, query) pairs that share a term; keep a min-heap per query.
    heaps: List[List[Tuple[float, str]]] = [[] for _ in qids]
    t0 = time.perf_counter()
    for i, (pid, _, toks) in enumerate(iter_corpus(corpus_path), start=1):
        tf = Counter(t for t in toks if t in term_queries)
        if tf:
            norm = K1 * (1.0 - B + B * len(toks) / avg_len)
            scores: Dict[int, float] = defaultdict(float)
            for term, f in tf.items():
                w = idf[term] * f * (K1 + 1.0) / (f + norm)
                for qi in term_queries[term]:
                    scores[qi] += w
            for qi, s in scores.items():
                h = heaps[qi]
                if len(h) < depth:
                    heapq.heappush(h, (s, pid))
                elif s > h[0][0]:
                    heapq.heapreplace(h, (s, pid))
        if i % log_every == 0:
            print(f"  pass 2: {i:,} passages ({time.perf_counter() - t0:.0f}s)", flush=True)
    return {qid: sorted(heaps[qi], reverse=True) for qi, qid in enumerate(qids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MS MARCO dev sample (BEIR format)")
    parser.add_argument("--source", default=SOURCE_DIR, help="Unzipped BEIR msmarco directory.")
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--depth", type=int, default=100, help="BM25 candidates kept per query.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    qrels = load_qrels(os.path.join(args.source, "qrels", "dev.tsv"))
    qids = sample_query_ids(qrels, args.n_queries, args.seed)
    wanted = set(qids)
    queries: Dict[str, str] = {}
    with open(os.path.join(args.source, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in wanted:
                queries[q["_id"]] = q["text"]
    qids = [q for q in qids if q in queries]
    print(f"Sampled {len(qids)} dev queries (seed {args.seed}); scoring BM25 over the full collection ...", flush=True)

    corpus_path = os.path.join(args.source, "corpus.jsonl")
    pools = bm25_pools(corpus_path, {q: queries[q] for q in qids}, args.depth)
    keep = {pid for pool in pools.values() for _, pid in pool}
    relevant = {pid for q in qids for pid in qrels[q]}
    keep |= relevant

    # Pass 3: write the pooled corpus.
    os.makedirs(os.path.join(args.output, "qrels"), exist_ok=True)
    written = 0
    with open(corpus_path, encoding="utf-8") as src, \
            open(os.path.join(args.output, "corpus.jsonl"), "w", encoding="utf-8") as out:
        for line in src:
            if json.loads(line)["_id"] in keep:
                out.write(line if line.endswith("\n") else line + "\n")
                written += 1
    with open(os.path.join(args.output, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q in qids:
            f.write(json.dumps({"_id": q, "text": queries[q]}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output, "qrels", "dev.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in qids:
            for pid, score in sorted(qrels[q].items()):
                f.write(f"{q}\t{pid}\t{score}\n")

    in_pool = sum(any(pid in {p for _, p in pools[q]} for pid in qrels[q]) for q in qids)
    info = {
        "source": "BEIR msmarco (MS MARCO passage v1), dev split", "n_queries": len(qids), "seed": args.seed,
        "bm25_depth": args.depth, "bm25": {"k1": K1, "b": B, "tokenizer": "retrieval.bm25_retriever.tokenize"},
        "corpus_passages": written, "judged_relevant_passages": len(relevant),
        "queries_with_a_relevant_passage_in_bm25_pool": in_pool,
    }
    with open(os.path.join(args.output, "sample_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
