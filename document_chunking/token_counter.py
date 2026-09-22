from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseTokenCounter(ABC):
    """Abstract interface for counting tokens in text."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text string."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert text into a list of token strings."""
        pass


class StandardTokenCounter(BaseTokenCounter):
    """Standard token counter using word and punctuation boundary parsing.
    
    Acts as a lightweight, deterministic proxy for subword BPE tokenization.
    Optionally delegates to tiktoken if available in the runtime environment.
    """

    def __init__(self, use_tiktoken_if_available: bool = True):
        self._tiktoken_encoder: Optional[Any] = None
        if use_tiktoken_if_available:
            try:
                import tiktoken  # type: ignore
                self._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            except (ImportError, Exception):
                self._tiktoken_encoder = None

        self._token_pattern = re.compile(r"\w+|[^\w\s]")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tiktoken_encoder is not None:
            return len(self._tiktoken_encoder.encode(text))
        return len(self._token_pattern.findall(text))

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return self._token_pattern.findall(text)
