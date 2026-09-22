from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Set

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator


class InternalConsistencyValidator(BaseValidator):
    """Validates date validity, document cross-references (related_documents), and structural relationships."""

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        all_doc_ids: Set[str] = {doc.document_id.strip() for doc in documents if doc.document_id}

        for doc in documents:
            doc_id = doc.document_id

            # 1. Effective date sanity
            if doc.effective_date:
                try:
                    dt = datetime.strptime(doc.effective_date.strip(), "%Y-%m-%d")
                    if dt.year < 2000 or dt.year > 2035:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.CONSISTENCY,
                                severity=Severity.WARNING,
                                document_id=doc_id,
                                field="effective_date",
                                message=f"Effective date '{doc.effective_date}' is outside realistic operational timeframe (2000-2035)",
                            )
                        )
                except ValueError:
                    pass  # Handled by SchemaValidator

            # 2. Dangling cross-references in related_documents
            for rel_id in doc.related_documents:
                cleaned_rel = rel_id.strip()
                # If it's a relationship code like REL-001, it refers to a world model relationship
                if cleaned_rel.startswith("REL-"):
                    continue

                if cleaned_rel not in all_doc_ids:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.CONSISTENCY,
                            severity=Severity.INFO,
                            document_id=doc_id,
                            field="related_documents",
                            message=f"Document references '{cleaned_rel}' in related_documents, which is not present in the current corpus",
                            details={"referenced_id": cleaned_rel},
                        )
                    )

            # 3. Self-referencing check
            if doc_id and doc_id in doc.related_documents:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONSISTENCY,
                        severity=Severity.WARNING,
                        document_id=doc_id,
                        field="related_documents",
                        message=f"Document '{doc_id}' contains a circular self-reference in related_documents",
                    )
                )

        return issues
