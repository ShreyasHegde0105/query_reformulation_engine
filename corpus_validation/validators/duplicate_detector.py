from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from corpus_validation.config import ValidationConfig
from corpus_validation.models import (
    Document,
    DuplicateGroup,
    NearDuplicatePair,
    Severity,
    ValidationCategory,
    ValidationIssue,
)
from corpus_validation.validators.base import BaseValidator


def normalize_content_for_hash(text: str) -> str:
    """Normalize text content for deterministic exact duplicate detection."""
    # Normalize CRLF to LF, collapse repeated spaces and tabs, strip lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Strip empty lines from outer edges and collapse internal repeated blank lines
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False
    return "\n".join(cleaned_lines).strip()


class DuplicateDetector(BaseValidator):
    """Detects exact duplicates via SHA-256 hashing and near-duplicates via token TF-IDF cosine similarity."""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.exact_duplicate_groups: List[DuplicateGroup] = []
        self.near_duplicate_pairs: List[NearDuplicatePair] = []

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        self.exact_duplicate_groups.clear()
        self.near_duplicate_pairs.clear()

        # ---------------------------------------------------------
        # 1. Exact Duplicate Detection (Normalized SHA-256 Hashing)
        # ---------------------------------------------------------
        hash_to_docs: Dict[str, List[str]] = defaultdict(list)
        doc_map: Dict[str, Document] = {}

        for doc in documents:
            doc_id = doc.document_id or f"UNKNOWN-{id(doc)}"
            doc_map[doc_id] = doc
            content = doc.content or ""
            if not content.strip():
                continue

            norm_content = normalize_content_for_hash(content)
            content_hash = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()
            hash_to_docs[content_hash].append(doc_id)

        dup_idx = 1
        for chash, doc_ids in hash_to_docs.items():
            if len(doc_ids) > 1:
                group_id = f"DUP-{dup_idx:03d}"
                canonical_id = doc_ids[0]
                dup_group = DuplicateGroup(
                    group_id=group_id,
                    documents=doc_ids,
                    content_hash=chash,
                    canonical_document_id=canonical_id,
                )
                self.exact_duplicate_groups.append(dup_group)
                dup_idx += 1

                for dup_doc_id in doc_ids[1:]:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.DUPLICATE,
                            severity=Severity.ERROR if self.config.fail_on_schema_error else Severity.WARNING,
                            document_id=dup_doc_id,
                            field="content",
                            message=f"Exact duplicate detected: identical content to canonical document '{canonical_id}' in group '{group_id}'",
                            details={"group_id": group_id, "canonical_id": canonical_id},
                        )
                    )

        # ---------------------------------------------------------
        # 2. Near-Duplicate Detection (Token TF-IDF & Inverted Index)
        # ---------------------------------------------------------
        if self.config.enable_near_duplicate_detection and len(documents) > 1:
            near_issues = self._detect_near_duplicates(documents)
            issues.extend(near_issues)

        return issues

    def _detect_near_duplicates(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        valid_docs = [d for d in documents if d.document_id and d.content and len(d.content.strip()) >= 50]
        if len(valid_docs) < 2:
            return issues

        # Tokenize and compute term frequencies
        doc_tokens: Dict[str, Counter[str]] = {}
        doc_norms: Dict[str, float] = {}
        doc_lengths: Dict[str, int] = {}
        inverted_index: Dict[str, List[str]] = defaultdict(list)

        stop_words = {
            "the", "and", "to", "of", "a", "in", "is", "that", "for", "on", "with",
            "as", "by", "at", "from", "this", "be", "are", "or", "an", "will", "all",
            "per", "shall", "under", "must", "may", "any", "which"
        }

        for doc in valid_docs:
            d_id = str(doc.document_id)
            content_text = str(doc.content)
            tokens = [
                w.lower()
                for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", content_text)
                if w.lower() not in stop_words
            ]
            if not tokens:
                continue

            tf = Counter(tokens)
            doc_tokens[d_id] = tf
            doc_lengths[d_id] = len(tokens)

            # Vector Euclidean norm
            norm = math.sqrt(sum(cnt * cnt for cnt in tf.values()))
            doc_norms[d_id] = norm if norm > 0 else 1.0

            # Build inverted index for rare/informative tokens to filter candidate pairs
            for token in set(tokens):
                inverted_index[token].append(d_id)

        # Candidate pair generation via inverted index (avoid O(N^2))
        candidate_pairs: Set[Tuple[str, str]] = set()
        for token, postings in inverted_index.items():
            # If token appears in too many docs (e.g. > 60%), skip indexing on it
            if len(postings) > max(10, len(valid_docs) * 0.60):
                continue
            for i in range(len(postings)):
                for j in range(i + 1, len(postings)):
                    d1, d2 = postings[i], postings[j]
                    if d1 == d2:
                        continue
                    pair = (min(d1, d2), max(d1, d2))
                    # Length filter: only compare docs with similar token lengths (within 35%)
                    l1, l2 = doc_lengths.get(d1, 0), doc_lengths.get(d2, 0)
                    if min(l1, l2) / max(l1, l2) >= 0.65:
                        candidate_pairs.add(pair)

        # Compute cosine similarity for candidate pairs
        threshold = self.config.near_duplicate_threshold
        seen_pairs: Set[Tuple[str, str]] = set()

        for d1, d2 in candidate_pairs:
            pair = (d1, d2)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            tf1, tf2 = doc_tokens[d1], doc_tokens[d2]
            common_words = set(tf1.keys()) & set(tf2.keys())
            if not common_words:
                continue

            dot_product = sum(tf1[w] * tf2[w] for w in common_words)
            similarity = dot_product / (doc_norms[d1] * doc_norms[d2])

            if similarity >= threshold:
                pair_record = NearDuplicatePair(
                    document_a=d1,
                    document_b=d2,
                    similarity=similarity,
                )
                self.near_duplicate_pairs.append(pair_record)

                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.DUPLICATE,
                        severity=Severity.WARNING,
                        document_id=d2,
                        field="content",
                        message=f"Near-duplicate document detected with '{d1}' (Cosine similarity: {similarity:.2%})",
                        details={"document_a": d1, "document_b": d2, "similarity": round(similarity, 4)},
                    )
                )

        return issues
