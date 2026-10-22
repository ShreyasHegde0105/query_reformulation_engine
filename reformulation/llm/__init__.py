"""LLM client package public API.

Exports the configuration object, client class and exception hierarchy.
"""

from .config import LLMConfig
from .client import LLMClient
from .fallback import FallbackLLMClient
from .exceptions import (
    LLMError,
    LLMConfigError,
    LLMPromptError,
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
)

__all__ = [
    "LLMConfig",
    "LLMClient",
    "FallbackLLMClient",
    "LLMError",
    "LLMConfigError",
    "LLMPromptError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMResponseError",
]
