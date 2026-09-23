"""LLM client configuration.

Provides :class:`LLMConfig` which loads configuration from environment variables
or can be instantiated directly. Validation ensures all required fields are
present and have sensible types.
"""

import os
from dataclasses import dataclass
from typing import Optional

from .exceptions import LLMConfigError

@dataclass(frozen=True)
class LLMConfig:
    """Configuration for :class:`~reformulation.llm.client.LLMClient`.

    Parameters are loaded from environment variables if not provided
    explicitly. Validation occurs at instantiation and raises
    :class:`LLMConfigError` on missing or malformed values.
    """

    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.0
    timeout_seconds: int = 30
    max_retries: int = 2

    @staticmethod
    def _parse_float(value: str, name: str) -> float:
        try:
            return float(value)
        except Exception as exc:
            raise LLMConfigError(f"Invalid float for {name}: {value}") from exc

    @staticmethod
    def _parse_int(value: str, name: str) -> int:
        try:
            return int(value)
        except Exception as exc:
            raise LLMConfigError(f"Invalid int for {name}: {value}") from exc

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create a configuration instance from environment variables.

        Required environment variables:
            * ``LLM_PROVIDER``
            * ``LLM_MODEL``
            * ``LLM_API_KEY``
            * ``LLM_BASE_URL``
        Optional variables with defaults are ``LLM_TEMPERATURE``,
        ``LLM_TIMEOUT_SECONDS`` and ``LLM_MAX_RETRIES``.
        """
        try:
            provider = os.getenv("LLM_PROVIDER")
            model = os.getenv("LLM_MODEL")
            api_key = os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL")
            if not all([provider, model, api_key, base_url]):
                raise LLMConfigError("Missing required LLM configuration environment variables")
            temperature = cls._parse_float(os.getenv("LLM_TEMPERATURE", "0.0"), "LLM_TEMPERATURE")
            timeout_seconds = cls._parse_int(os.getenv("LLM_TIMEOUT_SECONDS", "30"), "LLM_TIMEOUT_SECONDS")
            max_retries = cls._parse_int(os.getenv("LLM_MAX_RETRIES", "2"), "LLM_MAX_RETRIES")
            return cls(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        except LLMConfigError:
            raise
        except Exception as exc:
            raise LLMConfigError(f"Failed to load LLM configuration: {exc}") from exc
