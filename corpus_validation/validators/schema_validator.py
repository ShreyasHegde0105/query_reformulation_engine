from __future__ import annotations

import re
from typing import List

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator

DATE_REGEX = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


class SchemaValidator(BaseValidator):
    """Validates structural schema, field existence, and primitive/collection data types."""

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        for doc in documents:
            doc_id = doc.document_id

            # 1. Required string fields
            for req_field in self.config.required_document_fields:
                val = doc.raw.get(req_field)
                if val is None:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.SCHEMA,
                            severity=Severity.ERROR,
                            document_id=doc_id,
                            field=req_field,
                            message=f"Required field '{req_field}' is missing or null",
                        )
                    )
                elif not isinstance(val, str):
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.SCHEMA,
                            severity=Severity.ERROR,
                            document_id=doc_id,
                            field=req_field,
                            message=f"Field '{req_field}' must be a string, got {type(val).__name__}",
                        )
                    )
                elif not val.strip():
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.SCHEMA,
                            severity=Severity.ERROR,
                            document_id=doc_id,
                            field=req_field,
                            message=f"Required field '{req_field}' cannot be empty or whitespace only",
                        )
                    )

            # 2. Required list fields
            list_fields = [
                "audience",
                "keywords",
                "retrieval_signals",
                "systems",
                "roles",
                "related_documents",
            ]
            for lf in list_fields:
                if lf in doc.raw:
                    val = doc.raw.get(lf)
                    if not isinstance(val, list):
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.SCHEMA,
                                severity=Severity.ERROR,
                                document_id=doc_id,
                                field=lf,
                                message=f"Field '{lf}' must be a list, got {type(val).__name__}",
                            )
                        )
                    else:
                        for idx, item in enumerate(val):
                            if not isinstance(item, str):
                                issues.append(
                                    ValidationIssue(
                                        category=ValidationCategory.SCHEMA,
                                        severity=Severity.ERROR,
                                        document_id=doc_id,
                                        field=lf,
                                        message=f"Item at index {idx} in '{lf}' must be a string, got {type(item).__name__}",
                                    )
                                )

            # 3. Date format validation
            if "effective_date" in doc.raw:
                eff_date = doc.raw.get("effective_date")
                if eff_date is not None:
                    if not isinstance(eff_date, str):
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.SCHEMA,
                                severity=Severity.ERROR,
                                document_id=doc_id,
                                field="effective_date",
                                message=f"Field 'effective_date' must be a string, got {type(eff_date).__name__}",
                            )
                        )
                    elif not DATE_REGEX.match(eff_date.strip()):
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.SCHEMA,
                                severity=Severity.ERROR,
                                document_id=doc_id,
                                field="effective_date",
                                message=f"Invalid date format for 'effective_date': '{eff_date}'. Expected ISO format YYYY-MM-DD",
                            )
                        )

        return issues
