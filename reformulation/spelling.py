"""Corpus-vocabulary spelling correction ("did you mean" against the index).

A misspelled query word is corrected to a word that actually occurs in the
indexed documents, so the correction can only point at terms retrieval can
match. This replaces hand-written typo entries, which only ever fix the typos
someone thought to list.

Rules are deliberately conservative; a word is left untouched when:

* it already occurs in the corpus vocabulary, or is shorter than 4 letters;
* it is not purely alphabetic (codes, numbers, identifiers);
* no vocabulary word is one edit away (insert, delete, substitute or swap two
  adjacent letters), or the only candidates change the first letter. Two-edit
  corrections were tried and rejected: "revalution" became "resolution";
* two candidates tie at the best distance and neither is clearly more frequent
  (at least twice as common) – the correction would be a guess.

Usage::

    corrector = CorpusSpellCorrector.from_chunks_file("Data/chunks/chunks.jsonl")
    normalize_query("attendence shortage", spell_corrector=corrector)  # "Attendance shortage"
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

__all__ = ["CorpusSpellCorrector", "build_vocabulary"]

_WORD_RE = re.compile(r"[a-z]+")
_DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parents[1] / "Data" / "chunks" / "chunks.jsonl"


def build_vocabulary(texts: Iterable[str], min_length: int = 3) -> Counter:
    """Count lowercase alphabetic words of at least ``min_length`` letters."""
    vocab: Counter = Counter()
    for text in texts:
        vocab.update(w for w in _WORD_RE.findall(text.lower()) if len(w) >= min_length)
    return vocab


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Optimal string alignment distance (Damerau-Levenshtein with adjacent
    transpositions); returns ``limit + 1`` as soon as the distance exceeds ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class CorpusSpellCorrector:
    """Correct misspelled words to frequent words of the indexed corpus."""

    def __init__(self, vocabulary: Mapping[str, int], min_word_length: int = 4, max_distance: int = 1) -> None:
        self.vocabulary: Dict[str, int] = {w.lower(): int(c) for w, c in vocabulary.items() if c > 0}
        self.min_word_length = min_word_length
        self.max_distance = max_distance
        self._by_first_letter: Dict[str, list[str]] = {}
        for word in self.vocabulary:
            self._by_first_letter.setdefault(word[0], []).append(word)
        self._memo: Dict[str, Optional[str]] = {}

    @classmethod
    def from_chunks_file(cls, path: Optional[os.PathLike | str] = None) -> "CorpusSpellCorrector":
        """Build the vocabulary from ``chunks.jsonl`` (title, section and chunk text)."""
        path = Path(path or os.getenv("CHUNKS_PATH") or _DEFAULT_CHUNKS_PATH)
        texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    texts.extend(chunk.get(k) or "" for k in ("title", "section", "text"))
        return cls(build_vocabulary(texts))

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.vocabulary

    def correct(self, word: str) -> Optional[str]:
        """Return the corrected (lowercase) word, or ``None`` to leave ``word`` as is."""
        low = word.lower()
        if low in self._memo:
            return self._memo[low]
        result = self._correct(low)
        self._memo[low] = result
        return result

    def _correct(self, low: str) -> Optional[str]:
        if len(low) < self.min_word_length or not low.isalpha() or low in self.vocabulary:
            return None
        limit = self.max_distance
        best: list[tuple[int, int, str]] = []  # (distance, -frequency, word)
        for cand in self._by_first_letter.get(low[0], ()):
            if len(cand) < self.min_word_length:
                continue
            d = _edit_distance(low, cand, limit)
            if d <= limit:
                best.append((d, -self.vocabulary[cand], cand))
        if not best:
            return None
        best.sort()
        top_dist, top_negfreq, top_word = best[0]
        ties = [b for b in best[1:] if b[0] == top_dist]
        if ties and -top_negfreq < 2 * -ties[0][1]:
            return None  # no clear winner: do not guess
        return top_word
