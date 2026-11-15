"""Evaluation-only LLM relevance judge.

The judge grades (query, passage) pairs on the 0-3 rubric in :mod:`evaluation.judge.rubric`
for **offline evaluation only**. It is deliberately isolated from production:

* it is never imported by the API, the search pipeline, retrieval or the reranker
  (enforced by ``tests/test_component_separation.py``);
* it receives copies of passage text and returns new objects, so it cannot change
  search results;
* it is blind: passages are de-duplicated and ordered by a hash of (query id, chunk id),
  so neither the producing system nor the rank is visible to the model;
* its grades are a diagnostic. They are not ground truth, are never written into the
  label files, and never feed the production reranker.

Human annotations remain the primary relevance reference.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evaluation.judge.rubric import FIXED_INTENTS_BLOCK, PROMPT_TEMPLATE, PROMPT_VERSION, RUBRIC_TEXT, RUBRIC_VERSION
from evaluation.judge.schema import JudgmentValidationError, validate_judge_payload

logger = logging.getLogger("evaluation.judge")

__all__ = ["JudgeConfig", "JudgePassage", "PassageJudgment", "JudgeResult", "LLMRelevanceJudge"]

ANSWERABILITY = ("fully_answerable", "partially_answerable", "unanswerable")


@dataclasses.dataclass(frozen=True)
class JudgeConfig:
    """Configuration of the evaluation judge (independent of the reformulation LLM settings)."""

    provider: str = "groq"
    model: Optional[str] = None           # None = the provider preset's default model
    temperature: float = 0.0
    batch_size: int = 8                   # passages per LLM call
    max_passage_chars: int = 3000
    max_attempts: int = 2                 # a rejected response is retried once, then reported as rejected

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            provider=os.getenv("JUDGE_PROVIDER", "groq").strip() or "groq",
            model=(os.getenv("JUDGE_MODEL") or "").strip() or None,
            batch_size=int(os.getenv("JUDGE_BATCH_SIZE", "8")),
        )

    def build_client(self):
        """An :class:`LLMClient` for the judge's own provider key (never the fail-over chain)."""
        from reformulation.llm.client import LLMClient
        from reformulation.llm.config import provider_configs_from_env

        configs = [c for c in provider_configs_from_env() if c.provider == self.provider]
        if not configs:
            raise RuntimeError(f"No API key configured for judge provider {self.provider!r}")
        cfg = configs[0]
        cfg = dataclasses.replace(cfg, model=self.model or cfg.model,
                                  temperature=self.temperature if cfg.temperature is not None else None)
        return LLMClient(cfg)


@dataclasses.dataclass(frozen=True)
class JudgePassage:
    """Immutable copy of a passage given to the judge."""

    chunk_id: str
    title: str
    section: Optional[str]
    text: str

    @classmethod
    def from_any(cls, obj: Any) -> "JudgePassage":
        if isinstance(obj, Mapping):
            return cls(str(obj["chunk_id"]), str(obj.get("title") or ""), obj.get("section"),
                       str(obj.get("text") or obj.get("chunk_text") or ""))
        return cls(str(obj.chunk_id), str(getattr(obj, "title", "") or ""), getattr(obj, "section", None),
                   str(getattr(obj, "chunk_text", None) or getattr(obj, "text", "") or ""))


@dataclasses.dataclass(frozen=True)
class PassageJudgment:
    chunk_id: str
    grade: int
    intents_addressed: List[int]
    evidence: List[str]
    rationale: str


@dataclasses.dataclass
class JudgeResult:
    """Judge output for one query. ``status`` is ``"ok"`` or ``"rejected"``; rejected results carry no grades."""

    query_id: str
    query: str
    status: str
    query_intents: List[str]
    judgments: List[PassageJudgment]
    answerability: Optional[str]          # with respect to the judged passages only, not the corpus
    provider: str
    model: str
    rubric_version: str
    prompt_version: str
    timestamp: str
    passage_order: List[str]
    error: Optional[str] = None
    raw_responses: List[str] = dataclasses.field(default_factory=list)

    def grades(self) -> Dict[str, int]:
        return {j.chunk_id: j.grade for j in self.judgments}

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _parse_json(raw: str) -> Any:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    body = match.group(1) if match else raw
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise JudgmentValidationError("response contains no JSON object")
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError as exc:
        raise JudgmentValidationError(f"invalid JSON: {exc.msg}") from exc


def answerability_from(intents: Sequence[str], judgments: Sequence[PassageJudgment]) -> str:
    """Deterministic answerability from validated grades (relative to the judged passages)."""
    covered = {i for j in judgments if j.grade >= 2 for i in j.intents_addressed}
    if not covered:
        return "unanswerable"
    return "fully_answerable" if covered >= set(range(len(intents))) else "partially_answerable"


class LLMRelevanceJudge:
    """Grade passages for one query with an LLM, under a fixed rubric, blind to system and rank."""

    def __init__(self, llm_client: Any, config: Optional[JudgeConfig] = None,
                 clock=lambda: datetime.now(timezone.utc)) -> None:
        self.llm = llm_client
        self.config = config or JudgeConfig()
        self._clock = clock
        cfg = getattr(llm_client, "config", None)
        self.provider = getattr(cfg, "provider", self.config.provider)
        self.model = getattr(cfg, "model", self.config.model or "unknown")

    @staticmethod
    def blind_order(query_id: str, passages: Sequence[JudgePassage]) -> List[JudgePassage]:
        """De-duplicate by chunk id and order by a hash, so order reveals neither system nor rank."""
        unique = {p.chunk_id: p for p in passages}
        return sorted(unique.values(),
                      key=lambda p: hashlib.sha256(f"{query_id}\x1f{p.chunk_id}".encode()).hexdigest())

    def _passage_text(self, p: JudgePassage) -> str:
        header = "\n".join(x for x in (f"Title: {p.title}" if p.title else "",
                                       f"Section: {p.section}" if p.section else "") if x)
        body = p.text[: self.config.max_passage_chars]
        return f"{header}\n{body}" if header else body

    def _call(self, query: str, shown: Dict[str, str], intents: Optional[List[str]]) -> tuple[Dict[str, Any], List[str]]:
        passages_block = "\n\n".join(f"[{pid}]\n{text}" for pid, text in shown.items())
        intents_block = (FIXED_INTENTS_BLOCK.format(intents=json.dumps(intents)) if intents else "")
        prompt = PROMPT_TEMPLATE.format(rubric=RUBRIC_TEXT, query=query.replace('"', "'"),
                                        intents_block=intents_block, passages=passages_block)
        raws: List[str] = []
        last_err: Optional[Exception] = None
        for _ in range(self.config.max_attempts):
            raw = self.llm.generate(prompt)
            raws.append(raw)
            try:
                return validate_judge_payload(_parse_json(raw), shown, fixed_intents=intents), raws
            except JudgmentValidationError as exc:
                last_err = exc
                logger.warning("Judge response rejected: %s", exc)
        raise JudgmentValidationError(str(last_err)) from last_err

    def judge(self, query_id: str, query: str, passages: Sequence[Any]) -> JudgeResult:
        """Grade every passage for ``query``. Inputs are copied; nothing passed in is modified."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        ordered = self.blind_order(query_id, [JudgePassage.from_any(p) for p in passages])
        base = dict(query_id=query_id, query=query, provider=self.provider, model=self.model,
                    rubric_version=RUBRIC_VERSION, prompt_version=PROMPT_VERSION,
                    timestamp=self._clock().isoformat(), passage_order=[p.chunk_id for p in ordered])
        if not ordered:
            return JudgeResult(status="ok", query_intents=[], judgments=[], answerability="unanswerable", **base)

        intents: Optional[List[str]] = None
        judged: List[PassageJudgment] = []
        raws: List[str] = []
        try:
            for start in range(0, len(ordered), self.config.batch_size):
                batch = ordered[start:start + self.config.batch_size]
                ids = {f"P{i + 1}": p for i, p in enumerate(batch)}
                shown = {pid: self._passage_text(p) for pid, p in ids.items()}
                payload, call_raws = self._call(query, shown, intents)
                raws += call_raws
                intents = payload["query_intents"]
                for j in payload["judgments"]:
                    judged.append(PassageJudgment(chunk_id=ids[j["passage_id"]].chunk_id, grade=j["grade"],
                                                  intents_addressed=j["intents_addressed"],
                                                  evidence=j["evidence"], rationale=j["rationale"]))
        except Exception as exc:  # rejected or transport failure: report, never fabricate grades
            return JudgeResult(status="rejected", query_intents=intents or [], judgments=[], answerability=None,
                               error=f"{type(exc).__name__}: {exc}", raw_responses=raws, **base)
        return JudgeResult(status="ok", query_intents=intents or [], judgments=judged,
                           answerability=answerability_from(intents or [], judged), raw_responses=raws, **base)
