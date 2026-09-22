from __future__ import annotations

import re
from typing import List

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator

GENERIC_TITLES = {
    "document",
    "untitled",
    "new document",
    "sample",
    "test document",
    "page",
    "temp",
    "draft",
}

SECTION_HEADER_REGEX = re.compile(
    r"(^|\n)(\d+\.|\d+\.\d+|#+|[A-Z\s]{4,}:)\s+[A-Z]", re.MULTILINE
)


class QualityValidator(BaseValidator):
    """Validates enterprise search quality, structured sectioning, information density, and retrieval readiness."""

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        for doc in documents:
            doc_id = doc.document_id
            content = doc.content or ""
            title = doc.title or ""

            # 1. Title Quality Checks
            cleaned_title = title.strip().lower()
            if cleaned_title in GENERIC_TITLES:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.QUALITY,
                        severity=Severity.ERROR,
                        document_id=doc_id,
                        field="title",
                        message=f"Document title '{title}' is generic or empty placeholder",
                    )
                )
            elif len(title.split()) < 2:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.QUALITY,
                        severity=Severity.WARNING,
                        document_id=doc_id,
                        field="title",
                        message=f"Document title '{title}' is very short (fewer than 2 words); title should be descriptive for search indexing",
                    )
                )

            # 2. Structural Headings & Sectioning
            # Enterprise documents should have clear structural sections for chunking
            if not SECTION_HEADER_REGEX.search(content):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.QUALITY,
                        severity=Severity.INFO,
                        document_id=doc_id,
                        field="content",
                        message="Document does not contain identifiable structured numbered or markdown section headers (e.g. '1. PURPOSE', '## SCOPE'); section headers improve downstream chunking",
                    )
                )

            # 3. Entity & System Mentions
            # Searchable documents should reference institutional entities or systems
            has_systems = len(doc.systems) > 0
            has_roles = len(doc.roles) > 0
            if not has_systems and not has_roles:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.QUALITY,
                        severity=Severity.WARNING,
                        document_id=doc_id,
                        field="metadata",
                        message="Document lacks explicit references to enterprise systems and roles in metadata",
                    )
                )

            # 4. Search Retrieval Scenario Readiness
            # Verify that retrieval signals contain conversational, keyword, and typo variants
            signals = doc.retrieval_signals
            if signals:
                has_question = any("?" in s or s.lower().startswith(("how", "what", "where", "can", "when", "why")) for s in signals)
                if not has_question:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.QUALITY,
                            severity=Severity.INFO,
                            document_id=doc_id,
                            field="retrieval_signals",
                            message="Retrieval signals do not include natural language question formats (e.g. 'how to...', 'what is the...')",
                        )
                    )

        return issues
