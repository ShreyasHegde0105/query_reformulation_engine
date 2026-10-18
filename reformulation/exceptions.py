"""Custom exceptions for the query reformulation layer.

These exceptions are deliberately small – they only exist to give a clear
semantic signal when the LLM response cannot be parsed into the expected JSON
structure.
"""

class ReformulationError(RuntimeError):
    """Base class for all reformulation‑related errors."""
    pass

class ReformulationParseError(ReformulationError):
    """Raised when the LLM output cannot be parsed or does not match the
    expected schema (missing ``reformulations`` key, wrong type, etc.)."""
    pass

class QueryValidationError(ValueError):
    """Raised when the user's query itself is unacceptable (empty, too long,
    wrong type). The message is safe to show to the client."""
    pass

class RetrievalUnavailableError(RuntimeError):
    """Raised when every retrieval attempt for a search failed, so an empty
    result would wrongly look like "no matches"."""
    pass
