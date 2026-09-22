from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class SubsectionInfo:
    number: str
    title: str
    start_char: int = 0
    end_char: int = 0


@dataclasses.dataclass
class Section:
    title: str
    section_number: Optional[str]
    content: str
    subsection: Optional[str] = None
    subsections: List[SubsectionInfo] = dataclasses.field(default_factory=list)
    start_char: int = 0
    end_char: int = 0


@dataclasses.dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    department: str
    department_id: str
    document_type: str
    text: str
    chunk_index: int
    total_chunks: int
    token_count: int
    character_count: int
    section: Optional[str] = None
    section_number: Optional[str] = None
    subsection: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    version: Optional[str] = None
    effective_date: Optional[str] = None
    status: Optional[str] = None
    audience: List[str] = dataclasses.field(default_factory=list)
    keywords: List[str] = dataclasses.field(default_factory=list)
    retrieval_signals: List[str] = dataclasses.field(default_factory=list)
    systems: List[str] = dataclasses.field(default_factory=list)
    roles: List[str] = dataclasses.field(default_factory=list)
    related_documents: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "section": self.section,
            "section_number": self.section_number,
            "subsection": self.subsection,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "character_count": self.character_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "department": self.department,
            "department_id": self.department_id,
            "document_type": self.document_type,
            "version": self.version,
            "effective_date": self.effective_date,
            "status": self.status,
            "audience": self.audience,
            "keywords": self.keywords,
            "retrieval_signals": self.retrieval_signals,
            "systems": self.systems,
            "roles": self.roles,
            "related_documents": self.related_documents,
        }


@dataclasses.dataclass
class ChunkingStats:
    total_documents: int = 0
    total_chunks: int = 0
    average_chunks_per_document: float = 0.0
    average_chunk_tokens: float = 0.0
    median_chunk_tokens: float = 0.0
    min_chunk_tokens: int = 0
    max_chunk_tokens: int = 0
    sections_detected: int = 0
    documents_without_sections: int = 0
    recursive_splits: int = 0
    sentence_fallbacks: int = 0
    token_fallbacks: int = 0
    distributions: Dict[str, Dict[str, int]] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "average_chunks_per_document": round(self.average_chunks_per_document, 2),
            "average_chunk_tokens": round(self.average_chunk_tokens, 2),
            "median_chunk_tokens": round(self.median_chunk_tokens, 2),
            "min_chunk_tokens": self.min_chunk_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "sections_detected": self.sections_detected,
            "documents_without_sections": self.documents_without_sections,
            "recursive_splits": self.recursive_splits,
            "sentence_fallbacks": self.sentence_fallbacks,
            "token_fallbacks": self.token_fallbacks,
            "distributions": self.distributions,
        }
