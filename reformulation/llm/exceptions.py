# Exception hierarchy for the LLM client stub.

class LLMError(RuntimeError):
    """Base class for LLM client errors."""

class LLMAuthError(LLMError):
    """Raised when authentication with the LLM provider fails."""
    pass
