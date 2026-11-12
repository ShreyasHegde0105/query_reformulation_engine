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
    """Assembles diversity-aware RAG context from ranked retrieval candidates.

    Key capabilities:
    1. Document Diversity: Prevents a single document from consuming all context slots.
    2. Overlap Pruning: Deduplicates chunks with high token overlap (e.g. Jaccard >= 0.70).
    3. Query Aspect Coverage: Identifies multiple query facets (e.g. eligibility vs process)
       and ensures chunks covering distinct aspects are retained.
    4. Token Budgeting: Enforces a configurable maximum context token budget.
    5. Provenance Retention: Preserves document title, section, and original retrieval rank.
    """

    _ASPECT_STOPWORDS = frozenset({
        "what", "is", "are", "the", "for", "and", "or", "in", "of", "to", "how",
        "do", "i", "can", "a", "an", "on", "about", "with", "explain", "describe",
        "please", "give", "tell", "me", "rules", "policy", "guidelines"
    })

    def __init__(
        self,
        max_tokens: int = 2000,
        max_total_chunks: int = 6,
        max_chunks_per_doc: int = 2,
        redundancy_threshold: float = 0.70,
        max_chars_per_chunk: int = 1500,
    ) -> None:
        if max_tokens < 100:
            raise ValueError("max_tokens must be >= 100")
        if max_total_chunks < 1:
            raise ValueError("max_total_chunks must be >= 1")
        if max_chunks_per_doc < 1:
            raise ValueError("max_chunks_per_doc must be >= 1")

        self.max_tokens = max_tokens
        self.max_total_chunks = max_total_chunks
        self.max_chunks_per_doc = max_chunks_per_doc
        self.redundancy_threshold = redundancy_threshold
        self.max_chars_per_chunk = max_chars_per_chunk

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

            # Identify which query aspects this chunk addresses
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

        # Build formatted source block
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
            duplicate_chunks_pruned=len(pruned_chunk_ids),
            formatted_sources=formatted_sources,
        )
