from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from retrieval.models import RetrievedChunk

logger = logging.getLogger("retrieval.bm25")

_DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parents[1] / "Data" / "chunks" / "chunks.jsonl"

_STOP_WORDS = frozenset(
    "a an the in on at for to of and or is are was were be been being have has had "
    "do does did can could should would what which who whom how when where why i me "
    "my you your it its we our if from with by as this that these those am get not no".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Light, deterministic suffix stripping (plural / -ing / -ed).

    Deliberately conservative: it only needs to conflate common inflections
    (fee/fees, allotment/allotments, renew/renewed) so query and chunk terms meet.
    """
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    return word


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stop words, stem."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


class BM25Retriever:
    """In-process Okapi BM25 keyword retriever over the chunk corpus.

    Drop-in replacement for :class:`KeywordRetriever` (same ``retrieve`` contract,
    ``retriever_type="keyword"``). Unlike PostgreSQL ``ts_rank_cd`` it weights
    terms by inverse document frequency, so rare, discriminative terms
    (``revaluation``, ``CSE301``) dominate common ones (``student``, ``semester``).

    Each chunk is indexed over the concatenation of ``fields`` (by default
    title, section heading and chunk text), which lets a chunk match terms that
    only appear in its document title or section heading.

    Parameters
    ----------
    chunks:
        Chunk dicts in ``chunks.jsonl`` format. If omitted, they are loaded
        from ``chunks_path``.
    chunks_path:
        Path to ``chunks.jsonl``. Defaults to ``$CHUNKS_PATH`` or
        ``Data/chunks/chunks.jsonl`` under the project root.
    k1, b:
        Standard BM25 term-frequency saturation and length normalization.
    fields:
        Chunk keys concatenated into the indexed text.
    """

    def __init__(
        self,
        chunks: Optional[Sequence[Dict[str, Any]]] = None,
        chunks_path: Optional[os.PathLike | str] = None,
        k1: float = 1.2,
        b: float = 0.75,
        fields: Sequence[str] = ("title", "section", "text"),
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative.")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1.")
        if not fields:
            raise ValueError("fields must name at least one chunk key.")

        if chunks is None:
            path = Path(chunks_path or os.getenv("CHUNKS_PATH") or _DEFAULT_CHUNKS_PATH)
            chunks = self._load_chunks(path)

        self.k1 = float(k1)
        self.b = float(b)
        self.fields = tuple(fields)
        self._chunks: List[Dict[str, Any]] = list(chunks)
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    @staticmethod
    def _load_chunks(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _index_text(self, chunk: Dict[str, Any]) -> str:
        parts: List[str] = []
        for field in self.fields:
            value = chunk.get(field)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, Iterable):
                parts.extend(str(v) for v in value)
        return " ".join(parts)

    def _build_index(self) -> None:
        self._postings: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        self._doc_len: List[int] = []

        for idx, chunk in enumerate(self._chunks):
            terms = tokenize(self._index_text(chunk))
            self._doc_len.append(len(terms))
            for term, tf in Counter(terms).items():
                self._postings[term].append((idx, tf))

        n_docs = len(self._chunks)
        self._avg_len = (sum(self._doc_len) / n_docs) if n_docs else 0.0
        self._idf = {
            term: math.log(1.0 + (n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self._postings.items()
        }
        logger.info("BM25 index built: %d chunks, %d terms.", n_docs, len(self._postings))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        match_count: int = 20,
        filter_domain: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Return up to ``match_count`` chunks ordered by BM25 score (descending).

        Chunks with no query-term overlap are never returned.

        Raises:
            ValueError: If query or parameters are invalid.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if isinstance(match_count, bool) or not isinstance(match_count, int) or match_count < 1 or match_count > 100:
            raise ValueError("match_count must be an integer between 1 and 100.")

        clean_domain: Optional[str] = None
        if filter_domain is not None:
            if not isinstance(filter_domain, str):
                raise ValueError("filter_domain must be a string or None.")
            clean_domain = filter_domain.strip() or None

        scores: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for idx, tf in postings:
                norm = self.k1 * (1.0 - self.b + self.b * self._doc_len[idx] / self._avg_len)
                scores[idx] += idf * tf * (self.k1 + 1.0) / (tf + norm)

        if clean_domain is not None:
            scores = {i: s for i, s in scores.items() if self._chunks[i].get("department") == clean_domain}

        ranked = sorted(scores.items(), key=lambda item: (-item[1], self._chunks[item[0]]["chunk_id"]))
        return [self._to_retrieved(self._chunks[i], s) for i, s in ranked[:match_count]]

    @staticmethod
    def _to_retrieved(chunk: Dict[str, Any], score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(chunk["chunk_id"]),
            document_id=str(chunk["document_id"]),
            title=str(chunk.get("title") or chunk["document_id"]),
            domain=str(chunk.get("department") or ""),
            chunk_index=int(chunk.get("chunk_index", 0)),
            chunk_text=str(chunk.get("text", "")),
            section=chunk.get("section"),
            token_count=chunk.get("token_count"),
            score=float(score),
            metadata={
                "document_type": chunk.get("document_type"),
                "section_number": chunk.get("section_number"),
                "version": chunk.get("version"),
                "effective_date": chunk.get("effective_date"),
            },
            retriever_type="keyword",
        )
