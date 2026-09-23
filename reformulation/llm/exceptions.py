"""Exception hierarchy for the LLM client.

All public exceptions inherit from :class:`LLMError`.
"""

class LLMError(RuntimeError):
    """Base class for all LLM client related errors."""

class LLMConfigError(LLMError):
    """Raised when the configuration is missing or invalid."""

class LLMPromptError(LLMError):
    """Raised when the supplied prompt is not a non‑empty string."""

class LLMAuthError(LLMError):
    """Raised when authentication with the provider fails (HTTP 401)."""

class LLMRateLimitError(LLMError):
    """Raised when the provider returns a rate‑limit response (HTTP 429)."""

class LLMTimeoutError(LLMError):
    """Raised when the HTTP request times out."""

class LLMResponseError(LLMError):
    """Raised when the provider returns a malformed or unexpected response."""
