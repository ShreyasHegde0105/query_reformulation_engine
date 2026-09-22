from __future__ import annotations

import re
from collections import Counter
from typing import List

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator


class ContentValidator(BaseValidator):
    """Validates document text content for length, whitespace ratio, repetition, placeholders, and encoding."""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.placeholder_regexes = [
            re.compile(p, re.IGNORECASE) for p in config.placeholder_patterns
        ]

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        for doc in documents:
            doc_id = doc.document_id
            content = doc.content

            # 1. Null / Empty content check
            if content is None:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=Severity.ERROR,
                        document_id=doc_id,
                        field="content",
                        message="Document content is null",
                    )
                )
                continue

            content_str = str(content)
            trimmed = content_str.strip()

            if not trimmed:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=Severity.ERROR,
                        document_id=doc_id,
                        field="content",
                        message="Document content is empty or contains only whitespace",
                    )
                )
                continue

            char_count = len(trimmed)
            words = re.findall(r"\b\w+\b", trimmed)
            word_count = len(words)

            # 2. Length thresholds
            if char_count < 50:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=Severity.ERROR,
                        document_id=doc_id,
                        field="content",
                        message=f"Document content is critically short ({char_count} chars < 50 chars)",
                        details={"char_count": char_count, "min_required": 50},
                    )
                )
            elif char_count < self.config.min_content_length:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=Severity.WARNING,
                        document_id=doc_id,
                        field="content",
                        message=f"Document content length ({char_count} chars) is below recommended minimum ({self.config.min_content_length})",
                        details={"char_count": char_count, "recommended_min": self.config.min_content_length},
                    )
                )

            if char_count > self.config.max_content_length:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=Severity.WARNING,
                        document_id=doc_id,
                        field="content",
                        message=f"Document content length ({char_count} chars) exceeds maximum threshold ({self.config.max_content_length})",
                        details={"char_count": char_count, "max_threshold": self.config.max_content_length},
                    )
                )

            if word_count < self.config.min_word_count:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=Severity.WARNING,
                        document_id=doc_id,
                        field="content",
                        message=f"Word count ({word_count} words) is below recommended minimum ({self.config.min_word_count})",
                        details={"word_count": word_count, "min_words": self.config.min_word_count},
                    )
                )

            # 3. Excessive whitespace ratio
            whitespace_chars = sum(1 for c in content_str if c.isspace())
            ws_ratio = whitespace_chars / len(content_str) if content_str else 0
            if ws_ratio > self.config.max_whitespace_ratio:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=Severity.WARNING,
                        document_id=doc_id,
                        field="content",
                        message=f"Excessive whitespace ratio ({ws_ratio:.2%}) exceeds threshold ({self.config.max_whitespace_ratio:.2%})",
                        details={"whitespace_ratio": ws_ratio},
                    )
                )

            # 4. Encoding anomalies (null bytes, invalid control characters)
            if "\x00" in content_str or "\ufffd" in content_str:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=Severity.ERROR,
                        document_id=doc_id,
                        field="content",
                        message="Content contains illegal control characters or unicode replacement characters (\\ufffd / \\x00)",
                    )
                )

            # 5. Placeholder / boilerplate text detection
            for pattern in self.placeholder_regexes:
                match = pattern.search(trimmed)
                if match:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.CONTENT,
                            severity=Severity.WARNING,
                            document_id=doc_id,
                            field="content",
                            message=f"Detected placeholder or boilerplate pattern: '{match.group(0)}'",
                            details={"pattern": pattern.pattern},
                        )
                    )

            # 6. Repetition detection (repeated 3-grams)
            if len(words) >= 12:
                trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
                trigram_counts = Counter(trigrams)
                if trigram_counts:
                    most_common_tg, highest_count = trigram_counts.most_common(1)[0]
                    repeat_ratio = (highest_count * 3) / len(words)
                    if repeat_ratio > self.config.max_repeated_ngram_ratio:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.CONTENT,
                                severity=Severity.WARNING,
                                document_id=doc_id,
                                field="content",
                                message=f"High content repetition detected: phrase '{' '.join(most_common_tg)}' appears {highest_count} times ({repeat_ratio:.1%} of text)",
                                details={"phrase": " ".join(most_common_tg), "count": highest_count},
                            )
                        )

            # 7. Title / Content alignment heuristic
            if doc.title:
                title_words = set(re.findall(r"\b[A-Za-z]{3,}\b", doc.title.lower()))
                # ignore generic words
                title_words = {w for w in title_words if w not in {"the", "and", "for", "with", "form", "policy", "guidelines"}}
                if title_words:
                    content_lower = trimmed.lower()
                    overlap = [w for w in title_words if w in content_lower]
                    if not overlap:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.CONTENT,
                                severity=Severity.INFO,
                                document_id=doc_id,
                                field="content",
                                message=f"Low lexical alignment between title ('{doc.title}') and content body",
                            )
                        )

        return issues
