"""Intent-gated query router for enterprise search.

Classifies incoming queries into deterministic intent classes and routes them
to the appropriate retrieval strategy (e.g. direct retrieval vs. LLM reformulation).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parents[1] / "Data" / "chunks" / "chunks.jsonl"


class QueryIntent(str, Enum):
    """Semantic intent classification of an enterprise search query."""

    EXACT_IDENTIFIER = "EXACT_IDENTIFIER"
    COURSE_CODE = "COURSE_CODE"
    DOCUMENT_ID = "DOCUMENT_ID"
    EXACT_TITLE = "EXACT_TITLE"
    KEYWORD_LOOKUP = "KEYWORD_LOOKUP"
    NATURAL_LANGUAGE = "NATURAL_LANGUAGE"
    ABBREVIATION = "ABBREVIATION"
    PROCEDURAL_QUERY = "PROCEDURAL_QUERY"
    AMBIGUOUS = "AMBIGUOUS"


class RoutingAction(str, Enum):
    """Retrieval routing action to execute for a given intent."""

    DIRECT = "DIRECT"  # Bypass reformulation, use direct hybrid search on the normalized query
    REFORMULATE = "REFORMULATE"  # Use full LLM reformulation pipeline + multi-query hybrid fusion
    SAFE_FALLBACK = "SAFE_FALLBACK"  # Terse or ambiguous queries fall back to safe single-query retrieval


@dataclasses.dataclass(frozen=True)
class IntentRouteResult:
    """Diagnostic outcome of the intent routing classification."""

    query: str
    normalized_query: str
    intent: QueryIntent
    action: RoutingAction
    confidence: float
    matched_entity: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "intent": self.intent.value,
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "matched_entity": self.matched_entity,
            "reason": self.reason,
        }


class IntentRouter:
    """Deterministically classifies query intent and selects the optimal retrieval route."""

    # Course code: 2-4 letters followed by 3 digits (e.g. MEC204, ECE302, CSE 301, CS101)
    # Uses negative lookaround to prevent matching hyphenated document IDs like HSTL-REG-003.
    _COURSE_CODE_RE = re.compile(r"(?<![-A-Za-z0-9])([A-Za-z]{2,4})[- ]?(\d{3})(?![-A-Za-z0-9])")

    # Document ID: e.g. HSTL-REG-003, ACAD-NOT-002, POL-ATTEND, TRANS-SOP-001
    _DOC_ID_RE = re.compile(
        r"\b([A-Z]{2,4}-[A-Z0-9]{2,5}(?:-\d{2,4})?|POL-[A-Z0-9\-]+)\b",
        re.IGNORECASE,
    )

    # Specific identifiers: Form numbers, ticket numbers, standards (e.g. STU-FRM-003, GRV-2026-00417, ISBN-978, RFC-1918)
    _EXACT_IDENTIFIER_RE = re.compile(
        r"\b([A-Z]{2,4}-\d{4}-\d{3,5}|(?:STU|FIN|ADM|EXAM|IT|SEC)-FRM-\d{3}|(?:ISBN|RFC|ISO)-\d+)\b",
        re.IGNORECASE,
    )

    # Procedural indicators
    _PROCEDURE_PREFIX_RE = re.compile(
        r"^(how (?:do i|to|can i|should i)|steps? to|procedure (?:to|for)|process (?:to|for)|workflow (?:to|for)|guidelines? to)\b",
        re.IGNORECASE,
    )

    # Natural language indicators
    _NATURAL_LANG_PREFIX_RE = re.compile(
        r"^(who|what|where|when|why|which|can i|is it|are there|am i|do i|does|will|could i)\b",
        re.IGNORECASE,
    )

    # Common domain acronyms and abbreviations
    _ABBREV_TOKENS = frozenset({
        "gpa", "cgpa", "erp", "frro", "noc", "tc", "ta/da", "ta", "da", "od",
        "cr", "hod", "lms", "vpn", "ieee", "acm", "phd", "ug", "pg", "mtech",
        "btech", "mba", "msc", "bsc", "bca", "mca", "cc", "ipr", "see", "cie",
        "dor", "sla", "sop", "frm", "not", "gui", "pol", "reg", "reimb", "appl",
        "rpt", "req", "cert", "hrs", "yr", "dept", "intl", "eda", "vlsi",
    })

    _TITLE_STOPWORDS = frozenset({
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
        "with", "by", "about", "rules", "policy", "guidelines", "procedure",
        "sop", "manual", "handbook", "statute", "act", "ordinance",
    })

    def __init__(
        self,
        doc_ids: Optional[Set[str]] = None,
        titles: Optional[Set[str]] = None,
        sections: Optional[Set[str]] = None,
    ) -> None:
        self.doc_ids = {d.upper() for d in (doc_ids or set())}
        self.titles = titles or set()
        self.sections = sections or set()
        self._clauses = self._extract_clauses()

    @classmethod
    def from_chunks_file(cls, path: Optional[os.PathLike | str] = None) -> "IntentRouter":
        """Instantiate router populated from indexed corpus chunks."""
        file_path = Path(path or os.getenv("CHUNKS_PATH") or _DEFAULT_CHUNKS_PATH)
        doc_ids: Set[str] = set()
        titles: Set[str] = set()
        sections: Set[str] = set()
        if file_path.exists():
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        if c.get("document_id"):
                            doc_ids.add(c["document_id"])
                        if c.get("title"):
                            titles.add(c["title"])
                        if c.get("section"):
                            sections.add(c["section"])
        else:
            logger.warning("Chunks file not found at %s. Initializing router with rule-based heuristics only.", file_path)
        return cls(doc_ids=doc_ids, titles=titles, sections=sections)

    def _extract_clauses(self) -> Set[str]:
        """Extract canonical title and section clauses for high-precision title matching."""
        clauses: Set[str] = set()
        for text in list(self.titles) + list(self.sections):
            cleaned = re.sub(r"^\d+\.\s*", "", text).strip()
            if cleaned:
                clauses.add(cleaned)
            # Split compound titles (e.g. "Silence Hours and Room Inspections") into clauses
            for part in re.split(r"[,:;|\-–()]+", cleaned):
                part = part.strip()
                if len(part) >= 4:
                    clauses.add(part)
        return clauses

    def _clean_tokens(self, s: str) -> List[str]:
        cleaned = re.sub(r"^\d+\.\s*", "", s)
        return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", cleaned)]

    def _check_exact_title_match(self, query: str) -> Optional[Tuple[str, float]]:
        """Check if query is a verified exact or near-exact match for a document title or section.
        
        Requires high token overlap or exact clause matches. Does not trigger on generic
        short keyword queries (e.g. 'fee refund' or 'exam schedule').
        """
        q_tokens = self._clean_tokens(query)
        q_clean = " ".join(q_tokens)
        q_content = [t for t in q_tokens if t not in self._TITLE_STOPWORDS and len(t) > 2]

        if not q_tokens:
            return None

        # 1. Exact match with any full title or section
        for t in list(self.titles) + list(self.sections):
            t_clean = " ".join(self._clean_tokens(t))
            if q_clean == t_clean:
                return (t, 1.0)

        # 2. Exact match with a distinctive title/section clause (minimum 2 words)
        for c in self._clauses:
            c_clean = " ".join(self._clean_tokens(c))
            if q_clean == c_clean and len(q_tokens) >= 2:
                return (c, 0.95)

        # 3. For queries with >= 3 tokens, check high content overlap with clauses or titles
        if len(q_tokens) >= 3 and len(q_content) >= 2:
            q_set = set(q_content)
            for c in self._clauses:
                c_content = set(t for t in self._clean_tokens(c) if t not in self._TITLE_STOPWORDS and len(t) > 2)
                if not c_content:
                    continue
                # Jaccard overlap on content words
                jaccard = len(q_set.intersection(c_content)) / len(q_set.union(c_content))
                if jaccard >= 0.50 or (q_set.issubset(c_content) and len(q_set) / len(c_content) >= 0.60):
                    return (c, 0.90)

            # Document-level multi-word policy match (e.g. 'hostel curfew policy')
            for t in self.titles:
                t_content = set(t for t in self._clean_tokens(t) if t not in self._TITLE_STOPWORDS and len(t) > 2)
                if q_set.issubset(t_content) and len(q_set) >= 2:
                    if any(w in t.lower() for w in ["policy", "guidelines", "rules", "sop", "procedure"]):
                        return (t, 0.88)

        return None

    def route(self, query: str, normalized_query: Optional[str] = None) -> IntentRouteResult:
        """Classify query intent and return recommended retrieval action."""
        norm = normalized_query or query.strip()
        raw_clean = query.strip()

        # 1. Exact Identifier (Form numbers, grievance IDs, standards)
        ident_match = self._EXACT_IDENTIFIER_RE.search(raw_clean)
        if ident_match:
            ident = ident_match.group(1).upper()
            return IntentRouteResult(
                query=query,
                normalized_query=norm,
                intent=QueryIntent.EXACT_IDENTIFIER,
                action=RoutingAction.DIRECT,
                confidence=0.95,
                matched_entity=ident,
                reason=f"Matched exact identifier/form code '{ident}' - direct retrieval preserves specific form/record target",
            )

        # 2. Course Code
        course_match = self._COURSE_CODE_RE.search(raw_clean)
        if course_match:
            code = f"{course_match.group(1).upper()}{course_match.group(2)}"
            return IntentRouteResult(
                query=query,
                normalized_query=norm,
                intent=QueryIntent.COURSE_CODE,
                action=RoutingAction.DIRECT,
                confidence=0.98,
                matched_entity=code,
                reason=f"Matched academic course code '{code}' - direct retrieval prevents token splitting and syllabus dilution",
            )

        # 3. Document ID
        doc_id_match = self._DOC_ID_RE.search(raw_clean)
        if doc_id_match:
            candidate_id = doc_id_match.group(1).upper()
            if candidate_id in self.doc_ids or re.match(r"^[A-Z]{2,4}-[A-Z0-9]{2,5}-\d{2,4}$", candidate_id):
                return IntentRouteResult(
                    query=query,
                    normalized_query=norm,
                    intent=QueryIntent.DOCUMENT_ID,
                    action=RoutingAction.DIRECT,
                    confidence=1.0,
                    matched_entity=candidate_id,
                    reason=f"Matched document ID '{candidate_id}' - direct retrieval preserves exact target",
                )

        # 4. Procedural Query
        if self._PROCEDURE_PREFIX_RE.search(raw_clean):
            return IntentRouteResult(
                query=query,
                normalized_query=norm,
                intent=QueryIntent.PROCEDURAL_QUERY,
                action=RoutingAction.REFORMULATE,
                confidence=0.90,
                reason="Procedural how-to query - reformulation bridges workflow terminology",
            )

        # 5. Natural Language conversational questions
        if self._NATURAL_LANG_PREFIX_RE.search(raw_clean) and ("?" in raw_clean or len(raw_clean.split()) >= 5):
            return IntentRouteResult(
                query=query,
                normalized_query=norm,
                intent=QueryIntent.NATURAL_LANGUAGE,
                action=RoutingAction.REFORMULATE,
                confidence=0.88,
                reason="Conversational/natural language question - reformulation expands vocabulary to formal enterprise concepts",
            )

        # 6. Exact Title / Verified Section Match
        title_match = self._check_exact_title_match(raw_clean)
        if title_match:
            matched_phrase, conf = title_match
            return IntentRouteResult(
                query=query,
                normalized_query=norm,
                intent=QueryIntent.EXACT_TITLE,
                action=RoutingAction.DIRECT,
                confidence=conf,
                matched_entity=matched_phrase,
                reason=f"Matched verified document/section title '{matched_phrase}' (confidence {conf:.2f}) - direct retrieval avoids off-policy drift",
            )

        # 7. Abbreviation / Acronym Heavy Query
        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9/]+", raw_clean)]
        abbrev_hits = [t for t in tokens if t in self._ABBREV_TOKENS or (t.isupper() and len(t) >= 2)]
        if abbrev_hits:
            return IntentRouteResult(
                query=query,
                normalized_query=norm,
                intent=QueryIntent.ABBREVIATION,
                action=RoutingAction.REFORMULATE,
                confidence=0.85,
                matched_entity=", ".join(abbrev_hits),
                reason=f"Contains abbreviations/acronyms ({', '.join(abbrev_hits)}) - reformulation expands domain abbreviations",
            )

        # 8. Keyword Lookup
        if 2 <= len(tokens) <= 5:
            return IntentRouteResult(
                query=query,
                normalized_query=norm,
                intent=QueryIntent.KEYWORD_LOOKUP,
                action=RoutingAction.REFORMULATE,
                confidence=0.75,
                reason="Telegraphic keyword query - reformulation provides synonym exploration while preserving keywords",
            )

        # 9. Fallback: Ambiguous / Short query
        return IntentRouteResult(
            query=query,
            normalized_query=norm,
            intent=QueryIntent.AMBIGUOUS,
            action=RoutingAction.SAFE_FALLBACK,
            confidence=0.50,
            reason="Ambiguous or terse query - using safe fallback retrieval path",
        )
