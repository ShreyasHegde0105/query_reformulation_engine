"""Multi-document RAG context assembly component.

Selects, deduplicates, and structures retrieved document chunks into a coherent,
diversity-aware context window for grounded answer generation.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from retrieval.models import RetrievedChunk


@dataclasses.dataclass(frozen=True)
class ContextChunk:
    """A retrieved chunk selected for inclusion in the RAG prompt context."""

    chunk_id: str
    document_id: str
    title: str
    section: Optional[str]
    chunk_text: str
    original_rank: int
    score: float
    token_count: int
    covered_aspects: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "section": self.section,
            "original_rank": self.original_rank,
            "score": round(self.score, 4),
            "token_count": self.token_count,
            "covered_aspects": self.covered_aspects,
        }


@dataclasses.dataclass
class AssembledContext:
    """The assembled RAG context ready for LLM prompt injection."""

    query: str
    chunks: List[ContextChunk]
    total_tokens: int
    document_ids: List[str]
    query_aspects: List[str]
    covered_aspects: List[str]
    missing_aspects: List[str]
    duplicate_chunks_pruned: int
    formatted_sources: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "chunk_count": len(self.chunks),
            "total_tokens": self.total_tokens,
            "document_ids": self.document_ids,
            "unique_documents_count": len(self.document_ids),
            "query_aspects": self.query_aspects,
            "covered_aspects": self.covered_aspects,
            "missing_aspects": self.missing_aspects,
            "duplicate_chunks_pruned": self.duplicate_chunks_pruned,
        }


class RAGContextAssembler:
    """Assembles relevance-first, diversity-aware RAG context from ranked candidates.

    Key capabilities:
    1. Relevance-First Multi-Document Diversity: Dynamically balances document breadth
       and chunk depth. When multiple documents contribute necessary evidence, slots are
       reserved across documents. When a single document provides complete evidence,
       multiple essential chunks are retained without forcing irrelevant documents into context.
    2. Overlap Pruning: Deduplicates chunks with high token overlap (Jaccard >= threshold).
    3. Query Aspect Coverage: Identifies multiple query facets (e.g. eligibility vs process)
       and ensures chunks covering distinct aspects are retained.
    4. Minimal Sufficient Context: Selects the smallest useful collection of evidence that
       covers the user's information needs without polluting context with low-scoring junk.
    5. Token Budgeting: Enforces a configurable maximum context token budget.
    6. Provenance Retention: Preserves document title, section, and original retrieval rank.
    """

    _ASPECT_STOPWORDS = frozenset({
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could", "should",
        "would", "may", "might", "must", "shall", "will", "i", "my", "me",
        "we", "our", "us", "you", "your", "they", "them", "their", "it", "its",
        "this", "that", "these", "those", "for", "of", "in", "on", "at", "to",
        "from", "by", "with", "about", "into", "through", "during", "before",
        "after", "above", "below", "up", "down", "out", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "just", "don", "shouldn", "now", "tell", "explain", "describe", "find",
        "give", "show", "list", "need", "know", "please", "help", "rules", "policy",
        "guidelines"
    })

    def __init__(
        self,
        max_tokens: int = 2000,
        max_total_chunks: int = 6,
        max_chunks_per_doc: int = 2,
        max_dominant_doc_chunks: int = 4,
        redundancy_threshold: float = 0.70,
        max_chars_per_chunk: int = 1500,
        strategy: str = "relevance_first",
        dominant_score_margin: float = 5.2,
        lookahead_window: int = 8,
    ) -> None:
        if max_tokens < 100:
            raise ValueError("max_tokens must be >= 100")
        if max_total_chunks < 1:
            raise ValueError("max_total_chunks must be >= 1")
        if max_chunks_per_doc < 1:
            raise ValueError("max_chunks_per_doc must be >= 1")
        if max_dominant_doc_chunks < max_chunks_per_doc:
            raise ValueError("max_dominant_doc_chunks must be >= max_chunks_per_doc")
        if strategy not in ("relevance_first", "calibrated_relative", "diversity_capped", "legacy"):
            raise ValueError(f"Unknown context assembly strategy: {strategy!r}")

        self.max_tokens = max_tokens
        self.max_total_chunks = max_total_chunks
        self.max_chunks_per_doc = max_chunks_per_doc
        self.max_dominant_doc_chunks = max_dominant_doc_chunks
        self.redundancy_threshold = redundancy_threshold
        self.max_chars_per_chunk = max_chars_per_chunk
        self.strategy = strategy
        self.dominant_score_margin = dominant_score_margin
        self.lookahead_window = lookahead_window

    def extract_query_aspects(self, query: str) -> List[str]:
        """Extract key information facets or sub-questions from the query."""
        raw_parts = re.split(r",|;|\band\b|\bor\b|\bas well as\b", query.lower())
        aspects: List[str] = []
        for part in raw_parts:
            tokens = [w for w in re.findall(r"[a-z0-9]+", part) if w not in self._ASPECT_STOPWORDS and len(w) > 2]
            if tokens:
                aspects.append(" ".join(tokens))
        return aspects or [query.strip().lower()]

    def calculate_token_overlap(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity over word tokens between two passages."""
        toks1 = set(re.findall(r"[a-z0-9]+", text1.lower()))
        toks2 = set(re.findall(r"[a-z0-9]+", text2.lower()))
        if not toks1 or not toks2:
            return 0.0
        return len(toks1.intersection(toks2)) / float(len(toks1.union(toks2)))

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from character and word lengths."""
        return max(1, len(text.split()))

    def assemble_context(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
    ) -> AssembledContext:
        """Select and format the most diverse, relevant, non-redundant chunk set."""
        if not candidates:
            return AssembledContext(
                query=query,
                chunks=[],
                total_tokens=0,
                document_ids=[],
                query_aspects=self.extract_query_aspects(query),
                covered_aspects=[],
                missing_aspects=self.extract_query_aspects(query),
                duplicate_chunks_pruned=0,
                formatted_sources="",
            )

        if self.strategy == "legacy":
            return self._assemble_legacy(query, candidates)
        elif self.strategy == "diversity_capped":
            return self._assemble_diversity_capped(query, candidates)
        else:
            return self._assemble_relevance_first(query, candidates)

    def _assemble_legacy(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
    ) -> AssembledContext:
        """Naive top-K slicing without document diversity or redundancy pruning."""
        query_aspects = self.extract_query_aspects(query)
        selected_chunks: List[ContextChunk] = []
        covered_aspects_set: Set[str] = set()
        total_tokens = 0

        for rank, chunk in enumerate(candidates[: self.max_total_chunks], start=1):
            chunk_text = (chunk.chunk_text or "").strip()
            est_tokens = chunk.token_count if (chunk.token_count and chunk.token_count > 0) else self.estimate_tokens(chunk_text)
            if total_tokens + est_tokens > self.max_tokens and selected_chunks:
                continue

            chunk_lower = (f"{chunk.title} {chunk.section or ''} {chunk_text}").lower()
            chunk_aspects = [asp for asp in query_aspects if any(word in chunk_lower for word in asp.split())]
            covered_aspects_set.update(chunk_aspects)

            context_chunk = ContextChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id or "UNKNOWN_DOC",
                title=chunk.title or "Untitled Document",
                section=chunk.section,
                chunk_text=chunk_text[: self.max_chars_per_chunk],
                original_rank=rank,
                score=chunk.score,
                token_count=est_tokens,
                covered_aspects=chunk_aspects,
            )
            selected_chunks.append(context_chunk)
            total_tokens += est_tokens

        return self._build_assembled_context(query, selected_chunks, query_aspects, covered_aspects_set, 0, total_tokens)

    def _assemble_diversity_capped(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
    ) -> AssembledContext:
        """Rigid diversity-capped context assembly (Phase 2.4 behavior)."""
        query_aspects = self.extract_query_aspects(query)
        selected_chunks: List[ContextChunk] = []
        doc_chunk_counts: Dict[str, int] = {}
        covered_aspects_set: Set[str] = set()
        pruned_chunk_ids: Set[str] = set()
        total_tokens = 0

        for rank, chunk in enumerate(candidates, start=1):
            if len(selected_chunks) >= self.max_total_chunks:
                break

            doc_id = chunk.document_id or "UNKNOWN_DOC"
            current_doc_count = doc_chunk_counts.get(doc_id, 0)
            if current_doc_count >= self.max_chunks_per_doc:
                continue

            chunk_text = (chunk.chunk_text or "").strip()
            if not chunk_text:
                continue

            is_redundant = False
            for sel in selected_chunks:
                if self.calculate_token_overlap(chunk_text, sel.chunk_text) >= self.redundancy_threshold:
                    is_redundant = True
                    pruned_chunk_ids.add(chunk.chunk_id)
                    break
            if is_redundant:
                continue

            est_tokens = chunk.token_count if (chunk.token_count and chunk.token_count > 0) else self.estimate_tokens(chunk_text)
            if total_tokens + est_tokens > self.max_tokens and selected_chunks:
                continue

            chunk_lower = (f"{chunk.title} {chunk.section or ''} {chunk_text}").lower()
            chunk_aspects = [asp for asp in query_aspects if any(word in chunk_lower for word in asp.split())]
            covered_aspects_set.update(chunk_aspects)

            context_chunk = ContextChunk(
                chunk_id=chunk.chunk_id,
                document_id=doc_id,
                title=chunk.title or "Untitled Document",
                section=chunk.section,
                chunk_text=chunk_text[: self.max_chars_per_chunk],
                original_rank=rank,
                score=chunk.score,
                token_count=est_tokens,
                covered_aspects=chunk_aspects,
            )

            selected_chunks.append(context_chunk)
            doc_chunk_counts[doc_id] = current_doc_count + 1
            total_tokens += est_tokens

        return self._build_assembled_context(
            query, selected_chunks, query_aspects, covered_aspects_set, len(pruned_chunk_ids), total_tokens
        )

    def _assemble_relevance_first(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
    ) -> AssembledContext:
        """Relevance-first multi-document context assembly (Phase 2.4.1)."""
        query_aspects = self.extract_query_aspects(query)
        selected_chunks: List[ContextChunk] = []
        doc_chunk_counts: Dict[str, int] = {}
        covered_aspects_set: Set[str] = set()
        pruned_chunk_ids: Set[str] = set()
        total_tokens = 0

        # Extract scores safely from metadata or score attribute
        def _get_chunk_score(c: RetrievedChunk) -> float:
            if hasattr(c, "metadata") and c.metadata and "ce_score" in c.metadata:
                try:
                    return float(c.metadata["ce_score"])
                except (ValueError, TypeError):
                    pass
            return float(c.score) if c.score is not None else 0.0

        top_chunk = candidates[0]
        top_doc = top_chunk.document_id or "UNKNOWN_DOC"
        top_score = _get_chunk_score(top_chunk)

        # Detect competitive secondary documents in top candidates using pure relative margin
        competitive_docs = {top_doc}
        for c in candidates[1: self.lookahead_window]:
            score = _get_chunk_score(c)
            if (top_score - score) <= self.dominant_score_margin:
                doc_id = c.document_id or "UNKNOWN_DOC"
                competitive_docs.add(doc_id)

        is_multi_doc = len(competitive_docs) > 1

        # Adaptive budget: single-focus queries need at most max_dominant_doc_chunks,
        # while multi-doc queries use max_total_chunks with base cap per document.
        effective_max_total = self.max_total_chunks if is_multi_doc else min(self.max_dominant_doc_chunks, self.max_total_chunks)
        base_cap = self.max_chunks_per_doc if is_multi_doc else self.max_dominant_doc_chunks

        for rank, chunk in enumerate(candidates, start=1):
            if len(selected_chunks) >= effective_max_total:
                break

            doc_id = chunk.document_id or "UNKNOWN_DOC"
            count = doc_chunk_counts.get(doc_id, 0)
            score = _get_chunk_score(chunk)
            score_gap = top_score - score
            chunk_text = (chunk.chunk_text or "").strip()
            if not chunk_text:
                continue

            chunk_lower = (f"{chunk.title} {chunk.section or ''} {chunk_text}").lower()
            chunk_aspects = [asp for asp in query_aspects if any(word in chunk_lower for word in asp.split())]
            new_aspects = [asp for asp in chunk_aspects if asp not in covered_aspects_set]

            # Stop check for single-doc query: don't pollute with irrelevant docs if top doc has >= 3 chunks
            if not is_multi_doc and len(selected_chunks) >= 3 and doc_id != top_doc and not new_aspects:
                if score_gap > self.dominant_score_margin or rank > 4:
                    break

            # If single-doc query and chunk is from a secondary document, reject if score gap exceeds dominant margin
            if not is_multi_doc and doc_id != top_doc and not new_aspects:
                if score_gap > self.dominant_score_margin:
                    continue

            # Admission filter for non-competitive documents on multi-doc queries
            if is_multi_doc and doc_id not in competitive_docs:
                if not new_aspects:
                    continue
                elif score_gap > (self.dominant_score_margin + 2.5):
                    continue

            # Capacity check
            if count >= base_cap:
                if new_aspects:
                    pass
                elif is_multi_doc and doc_id == top_doc and count < 3 and score_gap <= 2.5:
                    pass
                elif is_multi_doc and count < 3 and score_gap <= 1.5:
                    pass
                else:
                    continue

            # Check overlap against already selected chunks
            is_redundant = False
            for sel in selected_chunks:
                if self.calculate_token_overlap(chunk_text, sel.chunk_text) >= self.redundancy_threshold:
                    is_redundant = True
                    pruned_chunk_ids.add(chunk.chunk_id)
                    break
            if is_redundant:
                continue

            # Check token budget
            est_tokens = chunk.token_count if (chunk.token_count and chunk.token_count > 0) else self.estimate_tokens(chunk_text)
            if total_tokens + est_tokens > self.max_tokens and selected_chunks:
                continue

            context_chunk = ContextChunk(
                chunk_id=chunk.chunk_id,
                document_id=doc_id,
                title=chunk.title or "Untitled Document",
                section=chunk.section,
                chunk_text=chunk_text[: self.max_chars_per_chunk],
                original_rank=rank,
                score=score,
                token_count=est_tokens,
                covered_aspects=chunk_aspects,
            )

            selected_chunks.append(context_chunk)
            doc_chunk_counts[doc_id] = count + 1
            covered_aspects_set.update(chunk_aspects)
            total_tokens += est_tokens

        return self._build_assembled_context(
            query, selected_chunks, query_aspects, covered_aspects_set, len(pruned_chunk_ids), total_tokens
        )

    def _build_assembled_context(
        self,
        query: str,
        selected_chunks: List[ContextChunk],
        query_aspects: List[str],
        covered_aspects_set: Set[str],
        duplicates_pruned: int,
        total_tokens: int,
    ) -> AssembledContext:
        """Helper to build formatted source block and AssembledContext dataclass."""
        formatted_sources_list = []
        for c in selected_chunks:
            sec_header = f" — {c.section}" if c.section else ""
            header = f"[{c.chunk_id}] {c.title}{sec_header}"
            formatted_sources_list.append(f"{header}\n{c.chunk_text}")

        formatted_sources = "\n\n".join(formatted_sources_list)
        unique_docs = sorted(list(set(c.document_id for c in selected_chunks)))
        covered_aspects_list = [asp for asp in query_aspects if asp in covered_aspects_set]
        missing_aspects_list = [asp for asp in query_aspects if asp not in covered_aspects_set]

        return AssembledContext(
            query=query,
            chunks=selected_chunks,
            total_tokens=total_tokens,
            document_ids=unique_docs,
            query_aspects=query_aspects,
            covered_aspects=covered_aspects_list,
            missing_aspects=missing_aspects_list,
            duplicate_chunks_pruned=duplicates_pruned,
            formatted_sources=formatted_sources,
        )
