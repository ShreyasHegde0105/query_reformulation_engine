"""Normalization utilities for query reformulation.

The public function :func:`normalize_query` prepares a raw query string for the LLM
prompt.  Normalisation is deliberately lightweight – it cleans up whitespace and
common punctuation spacing but does **not** alter case, numbers, acronyms or
technical terms, because those are important for preserving the original intent.
"""

from __future__ import annotations

import re
from typing import Any

def _ensure_str(query: Any) -> str:
    """Internal helper that guarantees the input is a ``str``.

    The validation step already checks the type, but this guard makes the
    function robust when called directly.
    """
    if not isinstance(query, str):
        raise TypeError("normalize_query expects a string input")
    return query


def normalize_query(query: Any) -> str:
    """Return a whitespace‑cleaned version of *query*.

    Steps performed:

    1. **Trim** leading and trailing whitespace.
    2. **Collapse** any run of spaces, tabs, newlines, carriage returns into a
       single space.
    3. **Remove** space that appears *before* punctuation characters such as
       ``.,;:!?``.
    4. Return the resulting string **unchanged** with respect to case, numbers,
       acronyms or technical terms.

    The function is deterministic and side‑effect free.
    """
    s = _ensure_str(query)
    # Step 1 – strip
    s = s.strip()
    # Step 2 – collapse whitespace
    s = re.sub(r"[\s]+", " ", s)
    # Step 3 – remove space before punctuation
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    return s
