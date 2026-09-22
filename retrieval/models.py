from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional


@dataclasses.dataclass
class RetrievedChunk:
    """Standardized representation of a retrieved document chunk.
    
    Acts as the common contract across vector retrieval, keyword retrieval,
    and downstream hybrid RRF fusion.
    """

    chunk_id: str
    document_id: str
    title: str
    domain: str
    chunk_index: int
    chunk_text: str
    section: Optional[str]
    token_count: Optional[int]
    score: float
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    retriever_type: str = "vector"

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval result to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "domain": self.domain,
            "chunk_index": self.chunk_index,
            "chunk_text": self.chunk_text,
            "section": self.section,
            "token_count": self.token_count,
            "score": self.score,
            "metadata": self.metadata,
            "retriever_type": self.retriever_type,
        }
