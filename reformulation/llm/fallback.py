"""Fail-over across several LLM API keys and providers.

:class:`FallbackLLMClient` has the same ``generate(prompt)`` interface as
:class:`~reformulation.llm.client.LLMClient`. It tries the configured keys in
order and moves on when one fails: a rejected key (HTTP 401) is dropped for the
rest of the session; a key that hits its quota, times out or errors is skipped
for :attr:`FallbackLLMClient.cooldown_seconds` and then tried again. Each switch
is logged, so the terminal shows which provider answered.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .client import LLMClient
from .config import LLMConfig, provider_configs_from_env
from .exceptions import LLMAuthError, LLMConfigError, LLMError, LLMPromptError, LLMRateLimitError, LLMTimeoutError

logger = logging.getLogger(__name__)


@dataclass
class _Key:
    label: str
    client: LLMClient
    retry_at: float = 0.0
    dead: bool = False


def _reason(exc: LLMError) -> str:
    if isinstance(exc, LLMRateLimitError):
        return "is out of quota (rate limited)"
    if isinstance(exc, LLMTimeoutError):
        return "timed out"
    return f"failed ({exc})"


class FallbackLLMClient:
    """Try each LLM key in turn until one answers."""

    cooldown_seconds = 60.0

    def __init__(
        self,
        clients: Sequence[LLMClient],
        labels: Optional[Sequence[str]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not clients:
            raise LLMConfigError("No LLM API keys configured (set e.g. GEMINI_API_KEY in .env)")
        labels = list(labels) if labels is not None else [c.config.provider for c in clients]
        self._keys = [_Key(label, client) for label, client in zip(labels, clients)]
        self._clock = clock
        self.active: Optional[str] = None

    @classmethod
    def from_env(cls, sleep_func: Optional[Callable[[float], None]] = None) -> "FallbackLLMClient":
        """All provider keys from the environment, after the explicit ``LLM_*`` endpoint if one is set."""
        configs: List[LLMConfig] = []
        explicit = LLMConfig.explicit_from_env()
        if explicit is not None:
            configs.append(explicit)
        configs += [c for c in provider_configs_from_env() if explicit is None or c.api_key != explicit.api_key]

        totals = Counter(c.provider for c in configs)
        seen: Counter = Counter()
        labels = []
        for c in configs:
            seen[c.provider] += 1
            labels.append(f"{c.provider} key {seen[c.provider]}" if totals[c.provider] > 1 else c.provider)
        return cls([LLMClient(c, sleep_func=sleep_func) for c in configs], labels)

    @property
    def labels(self) -> List[str]:
        return [k.label for k in self._keys]

    def generate(self, prompt: str) -> str:
        LLMClient._validate_prompt(prompt)
        live = [k for k in self._keys if not k.dead]
        if not live:
            raise LLMAuthError("Every configured LLM API key was rejected")
        now = self._clock()
        # Keys still cooling down are tried only when nothing else is left.
        ready = [k for k in live if k.retry_at <= now] or live

        last_exc: Optional[LLMError] = None
        for i, key in enumerate(ready):
            try:
                text = key.client.generate(prompt)
            except LLMPromptError:
                raise
            except LLMError as exc:
                last_exc = exc
                if isinstance(exc, LLMAuthError):
                    key.dead = True
                    reason = "rejected the API key"
                else:
                    key.retry_at = self._clock() + self.cooldown_seconds
                    reason = _reason(exc)
                nxt = ready[i + 1].label if i + 1 < len(ready) else None
                logger.warning("LLM %s %s; %s", key.label, reason,
                               f"switching to {nxt}" if nxt else "no other LLM keys left")
                continue
            if key.label != self.active:
                logger.warning("LLM answering with %s (%s)", key.label, key.client.config.model)
                self.active = key.label
            return text
        assert last_exc is not None
        raise last_exc
