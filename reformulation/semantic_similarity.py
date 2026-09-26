import logging
import re
from dataclasses import dataclass, asdict
from typing import List, Sequence, Callable, Optional, Set, Dict, Tuple, Any

import numpy as np

# Import the existing embedding encoder – it provides embed_query().
# The encoder returns a *normalized* vector (unit length) as per its implementation.
# For unit‑tested usage we allow injecting a custom callable to avoid loading the model.
try:
    from embedding.encoder import EmbeddingEncoder
except Exception:  # pragma: no cover – fallback for environments without the package
    EmbeddingEncoder = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class FilterDecision:
    """Diagnostic decision record for a candidate query processed by SemanticSimilarity."""

    query: str
    similarity: float
    kept: bool
    reason: str
    compared_with: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "similarity": round(self.similarity, 4),
            "kept": self.kept,
            "reason": self.reason,
            "compared_with": self.compared_with,
        }


class SemanticSimilarity:
    """Utility to compute semantic similarity between queries and filter redundancies.

    Combines dense semantic similarity with lexical diversity checking to remove
    genuinely redundant query reformulations (near-identical paraphrases, exact
    duplicates, word reorderings) while preserving reformulations that introduce
    novel keywords or terminology essential for downstream hybrid (dense + BM25)
    retrieval.

    The filtering algorithm is globally order-independent: for any candidate set Q,
    the resulting selected set of queries is invariant to input candidate ordering.

    Parameters
    ----------
    threshold:
        Cosine-similarity threshold above which two queries are evaluated for
        redundancy. If similarity is below this value, queries are considered
        distinct and never redundant. Must be in ``[-1, 1]``.
    embed_fn:
        Optional callable ``(str) -> Sequence[float]`` used to obtain the embedding.
        If omitted, a fresh ``EmbeddingEncoder`` instance is created and its
        ``embed_query`` method is used.
    lexical_threshold:
        Lexical overlap (Jaccard similarity) threshold above which semantically
        similar queries are considered redundant. Must be in ``[0, 1]``.
        Default is 0.75.
    strict_semantic_threshold:
        Cosine-similarity threshold above which queries are considered near-identical
        in embedding space (e.g. minor inflectional/plural variations).
        Default is 0.97.
    """

    _STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "should", "would", "what", "which",
        "who", "whom", "how", "when", "where", "why", "i", "me", "my", "you",
        "your", "it", "its", "we", "our", "if", "from", "with", "by", "as",
        "this", "that", "these", "those", "am", "get"
    })

    def __init__(
        self,
        threshold: float = 0.85,
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
        lexical_threshold: float = 0.75,
        strict_semantic_threshold: float = 0.97,
    ) -> None:
        if not isinstance(threshold, (float, int)):
            raise ValueError("threshold must be a numeric value")
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("threshold must be between -1 and 1 inclusive")
        self.threshold = float(threshold)

        if not isinstance(lexical_threshold, (float, int)):
            raise ValueError("lexical_threshold must be a numeric value")
        if lexical_threshold < 0.0 or lexical_threshold > 1.0:
            raise ValueError("lexical_threshold must be between 0 and 1 inclusive")
        self.lexical_threshold = float(lexical_threshold)

        if not isinstance(strict_semantic_threshold, (float, int)):
            raise ValueError("strict_semantic_threshold must be a numeric value")
        if strict_semantic_threshold < -1.0 or strict_semantic_threshold > 1.0:
            raise ValueError("strict_semantic_threshold must be between -1 and 1 inclusive")
        self.strict_semantic_threshold = float(strict_semantic_threshold)

        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            if EmbeddingEncoder is None:
                raise RuntimeError(
                    "EmbeddingEncoder could not be imported; provide embed_fn explicitly"
                )
            # Lazily initialise the encoder – model loading only happens when needed.
            self._encoder = EmbeddingEncoder()
            self._embed_fn = self._encoder.embed_query

    # ------------------------------------------------------------------
    # Core similarity logic
    # ------------------------------------------------------------------
    def _to_numpy(self, vec: Sequence[float]) -> np.ndarray:
        """Convert any sequence of floats to a 1‑D numpy array of float32.
        The method validates dimensionality consistency.
        """
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Embedding vector must be one‑dimensional")
        return arr

    def similarity(self, query_a: str, query_b: str) -> float:
        """Return the cosine similarity between two queries.

        The underlying embeddings are expected to be unit-normalized (the
        project's ``EmbeddingEncoder`` normalises them).
        """
        if not isinstance(query_a, str) or not isinstance(query_b, str):
            raise ValueError("Both queries must be strings")
        emb_a = self._to_numpy(self._embed_fn(query_a))
        emb_b = self._to_numpy(self._embed_fn(query_b))
        if emb_a.shape != emb_b.shape:
            raise ValueError("Embedding vectors have mismatched dimensions")
        if emb_a.shape[0] < 3:
            raise ValueError("Embedding vectors have insufficient dimensionality")
        dot = float(np.dot(emb_a, emb_b))
        return max(min(dot, 1.0), -1.0)

    def are_similar(self, query_a: str, query_b: str) -> bool:
        """Return True if the cosine similarity is greater than or equal to the threshold."""
        return self.similarity(query_a, query_b) >= self.threshold

    def _get_content_tokens(self, text: str) -> Set[str]:
        """Extract normalized informative tokens excluding standard stop words."""
        raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:%|[+-])?", text.lower())
        content = {
            t for t in raw_tokens
            if t not in self._STOP_WORDS and (len(t) > 1 or t.isdigit())
        }
        if not content and raw_tokens:
            content = set(raw_tokens)
        return content

    def _canonicalize_duplicates(
        self,
        queries: List[str],
        original_query: Optional[str] = None
    ) -> List[str]:
        """Group exact duplicates (case-insensitive & whitespace-stripped)
        and choose a deterministic canonical representative for each group.
        """
        groups: Dict[str, List[str]] = {}
        for q in queries:
            if not isinstance(q, str):
                continue
            cleaned = q.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in groups:
                groups[key] = []
            groups[key].append(cleaned)

        if not groups:
            return []

        orig_clean = original_query.strip() if (original_query and isinstance(original_query, str)) else None
        orig_key = orig_clean.lower() if orig_clean else None

        canonical_list: List[str] = []
        for key, variants in groups.items():
            if orig_key and key == orig_key and orig_clean:
                canonical_list.append(orig_clean)
                continue

            # Deterministic selection between casing variants:
            # 1. Prefer sentence-case (first letter uppercase, rest lowercase)
            # 2. Avoid all-caps
            # 3. Prefer capitalized initial
            # 4. Stable alphabetical tie-break
            def _casing_score(s: str) -> Tuple[int, int, int, str]:
                is_sentence = 1 if (len(s) > 1 and s[0].isupper() and s[1:].islower()) else 0
                not_all_caps = 1 if not s.isupper() else 0
                is_cap = 1 if s[0].isupper() else 0
                return (is_sentence, not_all_caps, is_cap, s)

            best_variant = max(variants, key=_casing_score)
            canonical_list.append(best_variant)

        return canonical_list

    def _candidate_rank_score(
        self,
        candidate: str,
        orig_key: Optional[str] = None
    ) -> Tuple[int, int, int, int]:
        """Compute an input-order-independent ranking score for candidate selection.
        
        Priority 1: User's original query (if provided and matches)
        Priority 2: Number of unique content tokens (higher lexical specificity)
        Priority 3: Number of informative tokens (>2 chars or digits)
        Priority 4: Conciseness (fewer filler words for equal token content)
        """
        cand_key = candidate.strip().lower()
        is_orig = 1 if (orig_key is not None and cand_key == orig_key) else 0
        tokens = self._get_content_tokens(candidate)
        informative_count = sum(1 for t in tokens if len(t) > 2 or t.isdigit())
        total_words = len(candidate.split())

        return (
            is_orig,
            len(tokens),
            informative_count,
            -total_words,
        )

    def check_redundancy(
        self,
        candidate: str,
        accepted: str,
        sim: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Determine whether candidate query is redundant with an already accepted query,
        returning (is_redundant, diagnostic_reason).

        A candidate query is redundant if:
        1. It is an exact case-insensitive string match, OR
        2. Semantic cosine similarity >= strict_semantic_threshold (near-identical vector), OR
        3. Semantic cosine similarity >= threshold AND candidate provides NO meaningful
           lexical novelty (content tokens are identical, subset, or Jaccard similarity
           >= lexical_threshold).
        """
        if not isinstance(candidate, str) or not isinstance(accepted, str):
            raise ValueError("Both queries must be strings")

        cand_clean = candidate.strip().lower()
        acc_clean = accepted.strip().lower()

        # 1. Exact string match
        if cand_clean == acc_clean:
            return True, f"Exact duplicate of '{accepted}'"

        # Compute cosine similarity if not provided
        if sim is None:
            sim = self.similarity(candidate, accepted)

        # 2. Below semantic similarity threshold -> not redundant
        if sim < self.threshold:
            return False, None

        # 3. Near-identical vector in embedding space
        if sim >= self.strict_semantic_threshold:
            return True, f"High semantic redundancy (cosine similarity {sim:.4f} >= {self.strict_semantic_threshold:.2f})"

        # 4. Token-level lexical comparison
        tokens_cand = self._get_content_tokens(candidate)
        tokens_acc = self._get_content_tokens(accepted)

        # Identical content tokens (e.g. stop-word variation or reordered words)
        if tokens_cand == tokens_acc:
            return True, f"Identical content tokens with '{accepted}'"

        # Candidate adds no novel content tokens (subset of accepted query)
        novel_tokens = tokens_cand - tokens_acc
        if not novel_tokens:
            return True, f"No novel keywords / subset of '{accepted}'"

        # High lexical overlap (Jaccard similarity >= lexical_threshold)
        union = tokens_cand | tokens_acc
        if union:
            jaccard = len(tokens_cand & tokens_acc) / len(union)
            if jaccard >= self.lexical_threshold:
                return True, f"High lexical overlap with '{accepted}' (Jaccard {jaccard:.2f} >= {self.lexical_threshold:.2f})"

        return False, None

    def is_redundant(
        self,
        candidate: str,
        accepted: str,
        sim: Optional[float] = None,
    ) -> bool:
        """Determine whether candidate query is redundant with an already accepted query."""
        redundant, _ = self.check_redundancy(candidate, accepted, sim=sim)
        return redundant

    # ------------------------------------------------------------------
    # Filtering API
    # ------------------------------------------------------------------
    def filter_redundant_with_diagnostics(
        self,
        queries: List[str],
        original_query: Optional[str] = None
    ) -> Tuple[List[str], List[FilterDecision]]:
        """Remove semantically and lexically redundant queries, returning both
        the filtered query list and structured per-candidate diagnostic decisions.

        The selection of representatives is globally order-independent:
        for any permutation of queries, the resulting selected set is identical.

        Parameters
        ----------
        queries:
            List of candidate queries to filter.
        original_query:
            Optional original user query. If present in a redundancy group,
            it is guaranteed to be retained as the representative.

        Returns
        -------
        Tuple[List[str], List[FilterDecision]]:
            (retained_queries, diagnostic_decisions_for_all_candidates)
        """
        if not queries:
            return [], []

        orig_clean = original_query.strip() if (original_query and isinstance(original_query, str)) else None
        orig_key = orig_clean.lower() if orig_clean else None

        # 1. Deduplicate exact string matches with deterministic canonical casing
        canonical_queries = self._canonicalize_duplicates(queries, original_query=original_query)
        if not canonical_queries:
            return [], []

        # 2. Sort candidates by deterministic content-based ranking (input-order invariant)
        sorted_candidates = sorted(
            canonical_queries,
            key=lambda c: (
                -self._candidate_rank_score(c, orig_key=orig_key)[0],
                -self._candidate_rank_score(c, orig_key=orig_key)[1],
                -self._candidate_rank_score(c, orig_key=orig_key)[2],
                -self._candidate_rank_score(c, orig_key=orig_key)[3],
                c.lower(),
            )
        )

        # Pre-cache embeddings
        cache: Dict[str, np.ndarray] = {}
        for c in sorted_candidates:
            cache[c] = self._to_numpy(self._embed_fn(c))

        # 3. Non-redundant representative selection (global order-independent)
        selected_representatives: List[str] = []
        candidate_decisions: Dict[str, FilterDecision] = {}

        for cand in sorted_candidates:
            cand_key = cand.lower()
            is_orig = bool(orig_key and cand_key == orig_key)
            cand_vec = cache[cand]

            if not selected_representatives:
                selected_representatives.append(cand)
                candidate_decisions[cand_key] = FilterDecision(
                    query=cand,
                    similarity=1.0,
                    kept=True,
                    reason="Original normalized query" if is_orig else "Primary query formulation",
                    compared_with=None,
                )
                continue

            redundant = False
            best_prev = None
            max_sim = -1.0
            redundancy_reason = None
            redundant_with = None

            for prev in selected_representatives:
                prev_vec = cache[prev]
                sim = float(np.dot(cand_vec, prev_vec))
                sim = max(min(sim, 1.0), -1.0)

                if sim > max_sim:
                    max_sim = sim
                    best_prev = prev

                is_red, r_reason = self.check_redundancy(cand, prev, sim=sim)
                if is_red:
                    redundant = True
                    redundancy_reason = r_reason
                    redundant_with = prev
                    logger.debug(
                        "Query '%s' considered redundant (similarity=%.3f) with '%s'",
                        cand,
                        sim,
                        prev,
                    )
                    break

            if redundant:
                candidate_decisions[cand_key] = FilterDecision(
                    query=cand,
                    similarity=sim,
                    kept=False,
                    reason=redundancy_reason or f"Redundant with '{redundant_with}'",
                    compared_with=redundant_with,
                )
            else:
                selected_representatives.append(cand)
                candidate_decisions[cand_key] = FilterDecision(
                    query=cand,
                    similarity=max_sim if max_sim >= 0 else 1.0,
                    kept=True,
                    reason="Original normalized query" if is_orig else "Semantically distinct / adds retrieval diversity",
                    compared_with=best_prev if not is_orig else None,
                )

        selected_keys: Set[str] = {q.lower() for q in selected_representatives}

        # 4. Presentation ordering:
        # Preserve caller's query sequence for selected candidates
        presentation_output: List[str] = []
        seen_emitted: Set[str] = set()

        for q in queries:
            if not isinstance(q, str):
                continue
            k = q.strip().lower()
            if k in selected_keys and k not in seen_emitted:
                seen_emitted.add(k)
                rep = next((r for r in selected_representatives if r.lower() == k), q.strip())
                presentation_output.append(rep)

        # 5. Build full diagnostics list in input order of queries
        all_diagnostics: List[FilterDecision] = []
        seen_diag: Set[str] = set()

        for q in queries:
            if not isinstance(q, str):
                continue
            cleaned = q.strip()
            if not cleaned:
                continue
            k = cleaned.lower()
            if k in seen_diag:
                first_rep = next((dec.query for dec in all_diagnostics if dec.query.lower() == k), cleaned)
                all_diagnostics.append(
                    FilterDecision(
                        query=cleaned,
                        similarity=1.0,
                        kept=False,
                        reason=f"Exact duplicate of '{first_rep}'",
                        compared_with=first_rep,
                    )
                )
            else:
                seen_diag.add(k)
                if k in candidate_decisions:
                    dec = candidate_decisions[k]
                    all_diagnostics.append(
                        FilterDecision(
                            query=cleaned,
                            similarity=dec.similarity,
                            kept=dec.kept,
                            reason=dec.reason,
                            compared_with=dec.compared_with,
                        )
                    )
                else:
                    all_diagnostics.append(
                        FilterDecision(
                            query=cleaned,
                            similarity=1.0,
                            kept=False,
                            reason="Filtered",
                            compared_with=None,
                        )
                    )

        return presentation_output, all_diagnostics

    def filter_redundant(
        self,
        queries: List[str],
        original_query: Optional[str] = None
    ) -> List[str]:
        """Remove semantically and lexically redundant queries.

        The selection of representatives is globally order-independent:
        for any permutation of queries, the resulting selected set is identical.

        Parameters
        ----------
        queries:
            List of candidate queries to filter.
        original_query:
            Optional original user query. If present in a redundancy group,
            it is guaranteed to be retained as the representative.
        """
        output, _ = self.filter_redundant_with_diagnostics(queries, original_query=original_query)
        return output

    # ------------------------------------------------------------------
    # Convenience static method
    # ------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Calculate cosine similarity for two raw vectors (unit‑norm not required)."""
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            raise ValueError("Zero‑norm vector cannot be used for cosine similarity")
        return float(np.dot(a, b) / (norm_a * norm_b))
