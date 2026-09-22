from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(str, Enum):
    SCHEMA = "schema"
    ID = "id"
    CONTENT = "content"
    DUPLICATE = "duplicate"
    METADATA = "metadata"
    ENTITY = "entity"
    CONSISTENCY = "consistency"
    QUALITY = "search_quality"
    COVERAGE = "coverage"
    GROUND_TRUTH = "ground_truth"


@dataclass
class ValidationIssue:
    category: ValidationCategory
    severity: Severity
    message: str
    document_id: Optional[str] = None
    field: Optional[str] = None
    details: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "category": self.category.value if isinstance(self.category, ValidationCategory) else str(self.category),
            "severity": self.severity.value if isinstance(self.severity, Severity) else str(self.severity),
            "message": self.message,
        }
        if self.document_id is not None:
            result["document_id"] = self.document_id
        if self.field is not None:
            result["field"] = self.field
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class DuplicateGroup:
    group_id: str
    documents: List[str]
    content_hash: str
    canonical_document_id: str
    type: str = "exact_duplicate"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duplicate_group": self.group_id,
            "documents": self.documents,
            "canonical_document_id": self.canonical_document_id,
            "content_hash": self.content_hash,
            "type": self.type,
        }


@dataclass
class NearDuplicatePair:
    document_a: str
    document_b: str
    similarity: float
    type: str = "near_duplicate"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_a": self.document_a,
            "document_b": self.document_b,
            "similarity": round(self.similarity, 4),
            "type": self.type,
        }


@dataclass
class Document:
    raw: Dict[str, Any]
    source_file: Optional[str] = None
    domain: Optional[str] = None

    @property
    def document_id(self) -> Optional[str]:
        val = self.raw.get("document_id")
        return str(val) if val is not None else None

    @property
    def title(self) -> Optional[str]:
        val = self.raw.get("title")
        return str(val) if val is not None else None

    @property
    def content(self) -> Optional[str]:
        val = self.raw.get("content")
        return str(val) if val is not None else None

    @property
    def department(self) -> Optional[str]:
        val = self.raw.get("department")
        return str(val) if val is not None else None

    @property
    def department_id(self) -> Optional[str]:
        val = self.raw.get("department_id")
        return str(val) if val is not None else None

    @property
    def document_type(self) -> Optional[str]:
        val = self.raw.get("document_type")
        return str(val) if val is not None else None

    @property
    def version(self) -> Optional[str]:
        val = self.raw.get("version")
        return str(val) if val is not None else None

    @property
    def effective_date(self) -> Optional[str]:
        val = self.raw.get("effective_date")
        return str(val) if val is not None else None

    @property
    def status(self) -> Optional[str]:
        val = self.raw.get("status")
        return str(val) if val is not None else None

    @property
    def audience(self) -> List[str]:
        val = self.raw.get("audience", [])
        return val if isinstance(val, list) else []

    @property
    def keywords(self) -> List[str]:
        val = self.raw.get("keywords", [])
        return val if isinstance(val, list) else []

    @property
    def retrieval_signals(self) -> List[str]:
        val = self.raw.get("retrieval_signals", [])
        return val if isinstance(val, list) else []

    @property
    def systems(self) -> List[str]:
        val = self.raw.get("systems", [])
        return val if isinstance(val, list) else []

    @property
    def roles(self) -> List[str]:
        val = self.raw.get("roles", [])
        return val if isinstance(val, list) else []

    @property
    def related_documents(self) -> List[str]:
        val = self.raw.get("related_documents", [])
        return val if isinstance(val, list) else []

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


@dataclass
class QualityMetrics:
    total_words: int = 0
    average_word_count: float = 0.0
    median_word_count: float = 0.0
    min_word_count: int = 0
    max_word_count: int = 0
    average_char_count: float = 0.0
    median_char_count: float = 0.0
    min_char_count: int = 0
    max_char_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_words": self.total_words,
            "average_word_count": round(self.average_word_count, 2),
            "median_word_count": round(self.median_word_count, 2),
            "min_word_count": self.min_word_count,
            "max_word_count": self.max_word_count,
            "average_char_count": round(self.average_char_count, 2),
            "median_char_count": round(self.median_char_count, 2),
            "min_char_count": self.min_char_count,
            "max_char_count": self.max_char_count,
        }


@dataclass
class ValidationSummary:
    total_documents: int = 0
    valid_documents: int = 0
    rejected_documents: int = 0
    warnings: int = 0
    errors: int = 0
    infos: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "valid_documents": self.valid_documents,
            "rejected_documents": self.rejected_documents,
            "warnings": self.warnings,
            "errors": self.errors,
            "infos": self.infos,
        }
