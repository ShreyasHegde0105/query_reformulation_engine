from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Set

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator


class IdValidator(BaseValidator):
    """Validates document identifiers for uniqueness, existence, and naming conventions."""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.id_regex = re.compile(config.id_pattern)
        self.stats: Dict[str, int] = {
            "total_documents": 0,
            "unique_document_ids": 0,
            "duplicate_document_ids": 0,
            "invalid_document_ids": 0,
        }

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        self.stats["total_documents"] = len(documents)

        all_ids: List[str] = []
        id_counts: Counter[str] = Counter()

        for doc in documents:
            doc_id = doc.document_id

            if not doc_id or not doc_id.strip():
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.ID,
                        severity=Severity.ERROR,
                        field="document_id",
                        message="Document is missing an identifier or document_id is empty",
                    )
                )
                self.stats["invalid_document_ids"] += 1
                continue

            cleaned_id = doc_id.strip()
            all_ids.append(cleaned_id)
            id_counts[cleaned_id] += 1

            # Format verification
            if not self.id_regex.match(cleaned_id):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.ID,
                        severity=Severity.WARNING,
                        document_id=cleaned_id,
                        field="document_id",
                        message=f"Document ID '{cleaned_id}' does not match conventional format '{self.config.id_pattern}'",
                    )
                )

        self.stats["unique_document_ids"] = len(id_counts)

        # Collision detection
        duplicates = {k: v for k, v in id_counts.items() if v > 1}
        self.stats["duplicate_document_ids"] = len(duplicates)

        for dup_id, count in duplicates.items():
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.ID,
                    severity=Severity.ERROR,
                    document_id=dup_id,
                    field="document_id",
                    message=f"Duplicate document ID detected: '{dup_id}' appears {count} times across corpus",
                    details={"count": count},
                )
            )

        return issues
