from __future__ import annotations

import re
from collections import Counter
from statistics import median
from typing import Any, Dict, List

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, QualityMetrics, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator


class CoverageValidator(BaseValidator):
    """Calculates corpus distributions, word-count metrics, and flags underrepresented domains."""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.distributions: Dict[str, Dict[str, int]] = {
            "departments": {},
            "department_ids": {},
            "document_types": {},
            "domains": {},
            "statuses": {},
            "systems": {},
            "roles": {},
        }
        self.quality_metrics = QualityMetrics()

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        dept_counter: Counter[str] = Counter()
        dept_id_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        domain_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        sys_counter: Counter[str] = Counter()
        role_counter: Counter[str] = Counter()

        word_counts: List[int] = []
        char_counts: List[int] = []

        for doc in documents:
            if doc.department:
                dept_counter[doc.department] += 1
            if doc.department_id:
                dept_id_counter[doc.department_id] += 1
            if doc.document_type:
                type_counter[doc.document_type] += 1
            if doc.domain:
                domain_counter[doc.domain] += 1
            if doc.status:
                status_counter[doc.status] += 1

            for s in doc.systems:
                if s:
                    sys_counter[s] += 1
            for r in doc.roles:
                if r:
                    role_counter[r] += 1

            content = doc.content or ""
            words = re.findall(r"\b\w+\b", content)
            word_counts.append(len(words))
            char_counts.append(len(content))

        self.distributions["departments"] = dict(dept_counter)
        self.distributions["department_ids"] = dict(dept_id_counter)
        self.distributions["document_types"] = dict(type_counter)
        self.distributions["domains"] = dict(domain_counter)
        self.distributions["statuses"] = dict(status_counter)
        self.distributions["systems"] = dict(sys_counter.most_common(25))
        self.distributions["roles"] = dict(role_counter.most_common(25))

        # Calculate metrics
        if word_counts:
            self.quality_metrics.total_words = sum(word_counts)
            self.quality_metrics.average_word_count = sum(word_counts) / len(word_counts)
            self.quality_metrics.median_word_count = float(median(word_counts))
            self.quality_metrics.min_word_count = min(word_counts)
            self.quality_metrics.max_word_count = max(word_counts)

            self.quality_metrics.average_char_count = sum(char_counts) / len(char_counts)
            self.quality_metrics.median_char_count = float(median(char_counts))
            self.quality_metrics.min_char_count = min(char_counts)
            self.quality_metrics.max_char_count = max(char_counts)

        # Coverage sanity checks
        if len(documents) > 20:
            for dept, count in dept_counter.items():
                if count < 2:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.COVERAGE,
                            severity=Severity.INFO,
                            field="department",
                            message=f"Department '{dept}' has only {count} document(s) in the corpus",
                            details={"department": dept, "count": count},
                        )
                    )

        return issues
