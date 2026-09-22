from __future__ import annotations

import re


class TextNormalizer:
    """Normalizes document text prior to structural parsing and chunking.
    
    Adheres strictly to data integrity rules:
    - Normalizes line endings (CRLF and CR -> LF)
    - Normalizes non-breaking and unusual Unicode whitespace to standard spaces
    - Strips trailing whitespace from lines
    - Collapses excessive blank lines (3+ newlines -> 2 newlines) to maintain paragraph structure
    - NEVER removes meaningful words, numbers, or section headers
    - NEVER modifies the original source document in memory or storage
    """

    # Unicode space characters (non-breaking spaces, thin spaces, ideographic space, etc.)
    _UNICODE_SPACES = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")

    @classmethod
    def normalize_for_chunking(cls, text: str) -> str:
        """Normalize content text solely for chunking purposes."""
        if not text:
            return ""

        # 1. Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Normalize Unicode spaces
        normalized = cls._UNICODE_SPACES.sub(" ", normalized)

        # 3. Strip trailing whitespace per line to avoid false regex mismatches
        lines = [re.sub(r"[ \t]+$", "", line) for line in normalized.split("\n")]
        normalized = "\n".join(lines)

        # 4. Collapse accidental excessive blank lines (preserve double newlines for paragraphs)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized.strip()
