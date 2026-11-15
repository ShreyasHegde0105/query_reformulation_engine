"""Schema validation for LLM judge output.

The judge's JSON is untrusted model output. Every judgment is checked before it is
accepted; anything that does not satisfy the schema, or whose evidence is not
actually present in the passage, is rejected rather than repaired.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["JudgmentValidationError", "validate_judge_payload", "normalize_for_match"]

MAX_INTENTS = 5
MAX_EVIDENCE_SPANS = 5
MIN_EVIDENCE_CHARS = 4


class JudgmentValidationError(ValueError):
    """The judge's output does not satisfy the schema or the rubric's evidence rules."""


def normalize_for_match(text: str) -> str:
    """Whitespace- and case-insensitive form used to check that evidence is quoted from a passage."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise JudgmentValidationError(msg)


def validate_judge_payload(
    payload: Any,
    passages: Mapping[str, str],
    fixed_intents: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate one judge response against the passages it was shown.

    Parameters
    ----------
    payload:
        Parsed JSON from the judge.
    passages:
        ``{passage_id: passage_text}`` exactly as shown to the judge.
    fixed_intents:
        If the intents were supplied to the judge, the response must repeat them unchanged.

    Returns the validated payload (a normalised copy). Raises
    :class:`JudgmentValidationError` on any violation.
    """
    _require(isinstance(payload, dict), "payload must be a JSON object")
    _require(set(payload) == {"query_intents", "judgments"},
             f"payload keys must be exactly query_intents, judgments (got {sorted(payload)})")

    intents = payload["query_intents"]
    _require(isinstance(intents, list) and 1 <= len(intents) <= MAX_INTENTS,
             f"query_intents must be a list of 1-{MAX_INTENTS} strings")
    _require(all(isinstance(i, str) and i.strip() for i in intents), "every intent must be a non-empty string")
    if fixed_intents is not None:
        _require([i.strip() for i in intents] == [i.strip() for i in fixed_intents],
                 "query_intents must repeat the supplied intents unchanged")

    judgments = payload["judgments"]
    _require(isinstance(judgments, list), "judgments must be a list")
    ids = [j.get("passage_id") if isinstance(j, dict) else None for j in judgments]
    _require(sorted(map(str, ids)) == sorted(passages), "judgments must cover every passage id exactly once")

    normalized_passages = {pid: normalize_for_match(text) for pid, text in passages.items()}
    clean: List[Dict[str, Any]] = []
    for j in judgments:
        pid = j["passage_id"]
        _require(set(j) == {"passage_id", "grade", "intents_addressed", "evidence", "rationale"},
                 f"{pid}: judgment keys must be passage_id, grade, intents_addressed, evidence, rationale")
        grade = j["grade"]
        _require(isinstance(grade, int) and not isinstance(grade, bool) and 0 <= grade <= 3,
                 f"{pid}: grade must be an integer 0-3")
        addressed = j["intents_addressed"]
        _require(isinstance(addressed, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in addressed),
                 f"{pid}: intents_addressed must be a list of integers")
        _require(all(0 <= i < len(intents) for i in addressed), f"{pid}: intents_addressed index out of range")
        _require(len(set(addressed)) == len(addressed), f"{pid}: duplicate intent indexes")
        evidence = j["evidence"]
        _require(isinstance(evidence, list) and all(isinstance(e, str) for e in evidence),
                 f"{pid}: evidence must be a list of strings")
        _require(isinstance(j["rationale"], str), f"{pid}: rationale must be a string")

        if grade == 0:
            _require(not evidence and not addressed, f"{pid}: grade 0 must have no evidence and no intents")
        else:
            _require(1 <= len(evidence) <= MAX_EVIDENCE_SPANS,
                     f"{pid}: grades 1-3 need 1-{MAX_EVIDENCE_SPANS} evidence spans")
            for span in evidence:
                norm = normalize_for_match(span)
                _require(len(norm) >= MIN_EVIDENCE_CHARS, f"{pid}: evidence span too short: {span!r}")
                _require(norm in normalized_passages[pid], f"{pid}: evidence not found in passage: {span[:60]!r}")
        if grade >= 2:
            _require(bool(addressed), f"{pid}: grades 2-3 must name the intents they address")

        clean.append({"passage_id": pid, "grade": grade, "intents_addressed": list(addressed),
                      "evidence": list(evidence), "rationale": j["rationale"].strip()})
    return {"query_intents": [i.strip() for i in intents], "judgments": clean}
