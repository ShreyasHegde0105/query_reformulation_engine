"""Organisation glossary for vocabulary-aware query reformulation.

Users describe things in their own words ("websites blocked on campus
internet"); documents use the organisation's terms ("web content filtering",
"Acceptable Use Policy"). A generic LLM does not know those terms, so its
rewrites are paraphrases the dense retriever already understands. The glossary
gives the reformulation prompt the organisation's own vocabulary: department
names, system and portal names, and the document catalogue (titles).

Result (2026-10-20, tuning set, gemini-3.5-flash-lite): **rejected**. Recall@10
fell from 0.87 (generic prompt) to 0.78 and NDCG@10 from 0.646 to 0.562
(``evaluation/results/tune_glossary_report.md``). With the catalogue in the
prompt the LLM guesses which document to return and copies its title ("academic
transcript" -> Bonafide certificate titles) instead of restating the user's
specific need; a title query also matches every chunk of that document, which
pushes the answering chunk down. Production keeps the generic prompt; this
module stays as an opt-in experiment (``--glossary`` in
``evaluation/build_reformulation_cache.py``).

Only catalogue-level metadata is used. ``keywords`` and ``retrieval_signals``
are deliberately excluded: they were written alongside the documents and
include example user queries, which would leak the answers into the prompt.

Usage::

    glossary = load_glossary()                      # Data/validated/validated_documents.json
    reformulator = QueryReformulator(llm, glossary=glossary)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

__all__ = ["build_glossary", "load_glossary"]

_DEFAULT_DOCUMENTS_PATH = Path(__file__).resolve().parents[1] / "Data" / "validated" / "validated_documents.json"


def _as_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value or [] if str(v).strip()]


def _unique(values: Iterable[str]) -> List[str]:
    """Sorted, case-insensitively de-duplicated values (first spelling wins)."""
    seen: Dict[str, str] = {}
    for v in values:
        seen.setdefault(v.strip().lower(), v.strip())
    return sorted(seen.values(), key=str.lower)


def build_glossary(documents: Iterable[Dict[str, Any]]) -> str:
    """Render departments, systems and document titles as a compact prompt section."""
    documents = list(documents)
    departments = _unique(d.get("department") or "" for d in documents if d.get("department"))
    systems = _unique(s for d in documents for s in _as_list(d.get("systems")))
    titles = _unique(d.get("title") or "" for d in documents if d.get("title"))

    lines = []
    if departments:
        lines.append("Departments: " + "; ".join(departments))
    if systems:
        lines.append("Systems and portals: " + "; ".join(systems))
    if titles:
        lines.append("Document titles:")
        lines.extend(f"- {t}" for t in titles)
    return "\n".join(lines)


def load_glossary(path: Optional[Union[str, Path]] = None) -> str:
    """Build the glossary from a validated-documents file (``{"documents": [...]}`` or a list)."""
    with open(path or _DEFAULT_DOCUMENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    documents = data.get("documents", []) if isinstance(data, dict) else data
    return build_glossary(documents)
