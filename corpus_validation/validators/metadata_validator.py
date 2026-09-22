from __future__ import annotations

import re
from typing import List

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator

VERSION_REGEX = re.compile(r"^\d+(\.\d+){1,2}$")


class MetadataValidator(BaseValidator):
    """Validates enterprise metadata fields, controlled vocabularies, enums, versions, and lists."""

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        for doc in documents:
            doc_id = doc.document_id

            # 1. Required metadata fields existence
            for mf in self.config.required_metadata_fields:
                if mf not in doc.raw or doc.raw.get(mf) is None:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.ERROR,
                            document_id=doc_id,
                            field=mf,
                            message=f"Required metadata field '{mf}' is missing or null",
                        )
                    )

            # 2. Status enum validation
            status = doc.status
            if status is not None:
                if status not in self.config.allowed_statuses:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.ERROR,
                            document_id=doc_id,
                            field="status",
                            message=f"Status '{status}' is not in allowed status vocabulary {self.config.allowed_statuses}",
                            details={"status": status, "allowed": self.config.allowed_statuses},
                        )
                    )

            # 3. Version format check
            version = doc.version
            if version is not None:
                if not VERSION_REGEX.match(version.strip()):
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.WARNING,
                            document_id=doc_id,
                            field="version",
                            message=f"Version '{version}' does not follow semantic or decimal numbering convention (e.g. 1.0, 2.4)",
                            details={"version": version},
                        )
                    )

            # 4. Collection non-emptiness checks
            # Audience
            if "audience" in doc.raw:
                aud = doc.audience
                if not aud:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.WARNING,
                            document_id=doc_id,
                            field="audience",
                            message="Audience list is empty; institutional documents should specify target audience",
                        )
                    )

            # Keywords
            if "keywords" in doc.raw:
                kws = doc.keywords
                if not kws:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.WARNING,
                            document_id=doc_id,
                            field="keywords",
                            message="Keywords list is empty; searchable documents should specify domain keywords",
                        )
                    )
                elif len(kws) < 3:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.INFO,
                            document_id=doc_id,
                            field="keywords",
                            message=f"Keywords list has only {len(kws)} items (recommended: at least 3)",
                        )
                    )

            # Retrieval signals
            if "retrieval_signals" in doc.raw:
                rs = doc.retrieval_signals
                if not rs:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.WARNING,
                            document_id=doc_id,
                            field="retrieval_signals",
                            message="Retrieval signals list is empty; necessary for search scenario validation",
                        )
                    )
                elif len(rs) < 3:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.INFO,
                            document_id=doc_id,
                            field="retrieval_signals",
                            message=f"Retrieval signals list has only {len(rs)} items (recommended: at least 3)",
                        )
                    )

            # 5. Department ID format
            dept_id = doc.department_id
            if dept_id:
                if not dept_id.startswith("DEPT-"):
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.METADATA,
                            severity=Severity.WARNING,
                            document_id=doc_id,
                            field="department_id",
                            message=f"Department ID '{dept_id}' does not follow the standard 'DEPT-*' prefix format",
                        )
                    )

        return issues
