"""Grounded answer generation over retrieved chunks (RAG).

The LLM sees only the top retrieved chunks, must answer from them, and must
cite the chunk IDs it used. Citations are validated against the chunks that
were actually provided, so the model cannot cite something it was not shown.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Dict, List, Sequence

from retrieval.models import RetrievedChunk

from .exceptions import ReformulationParseError
from .llm.client import LLMClient
from .llm.fallback import FallbackLLMClient

__all__ = ["Answer", "AnswerGenerator"]


@dataclasses.dataclass
class Answer:
    """LLM answer with the chunk IDs it cites."""

    text: str
    citations: List[str]
    answerable: bool

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class AnswerGenerator:
    """Answer a question from retrieved chunks, with validated citations."""

    _NOT_FOUND = "The available documents do not answer this question."

    _PROMPT_TEMPLATE = (
        "You answer questions about university policies using ONLY the numbered sources below. "
        "Do not use outside knowledge. If the sources do not contain the answer, say so.\n\n"
        "Return a JSON object:\n"
        '{{\n  "answerable": true or false,\n'
        '  "answer": "concise answer; include exact figures, deadlines and conditions from the sources",\n'
        '  "citations": ["chunk_id", ...]\n}}\n'
        "`citations` must list the source IDs the answer relies on, taken exactly from the headers below.\n\n"
        "Question: {question}\n\n"
        "Sources:\n{sources}"
    )

    def __init__(self, llm_client: LLMClient, max_chunks: int = 5, max_chars_per_chunk: int = 2000):
        if not isinstance(llm_client, (LLMClient, FallbackLLMClient)):
            raise TypeError("llm_client must be an LLMClient or FallbackLLMClient")
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.llm_client = llm_client
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk

    def build_prompt(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        sources = []
        for chunk in chunks[: self.max_chunks]:
            header = f"[{chunk.chunk_id}] {chunk.title}" + (f" — {chunk.section}" if chunk.section else "")
            sources.append(f"{header}\n{chunk.chunk_text[: self.max_chars_per_chunk]}")
        return self._PROMPT_TEMPLATE.format(question=question.strip(), sources="\n\n".join(sources))

    def generate(self, question: str, chunks: Sequence[RetrievedChunk]) -> Answer:
        """Answer ``question`` from ``chunks``.

        Raises:
            ValueError: if the question is blank.
            ReformulationParseError: if the LLM response is not the expected JSON.
            LLMError (and subclasses): on LLM transport failures.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        provided = [c.chunk_id for c in chunks[: self.max_chunks]]
        if not provided:
            return Answer(text=self._NOT_FOUND, citations=[], answerable=False)

        raw = self.llm_client.generate(self.build_prompt(question, chunks))
        payload = self._parse(raw)

        answerable = payload.get("answerable")
        text = payload.get("answer")
        citations = payload.get("citations", [])
        if not isinstance(answerable, bool) or not isinstance(text, str) or not isinstance(citations, list):
            raise ReformulationParseError("Answer JSON must contain boolean 'answerable', string 'answer', list 'citations'")

        allowed = set(provided)
        valid = [c for c in dict.fromkeys(citations) if isinstance(c, str) and c in allowed]
        if not answerable or not text.strip():
            return Answer(text=text.strip() or self._NOT_FOUND, citations=[], answerable=False)
        if not valid:
            # An "answer" that cites none of the provided sources is not grounded.
            return Answer(text=self._NOT_FOUND, citations=[], answerable=False)
        return Answer(text=text.strip(), citations=valid, answerable=True)

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
        body = match.group(1).strip() if match else raw.strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ReformulationParseError(f"LLM answer is not valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ReformulationParseError("LLM answer JSON must be an object")
        return payload
