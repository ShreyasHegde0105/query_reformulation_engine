"""Validator utilities for query reformulation.

This module provides a single public function :func:`validate_query` that checks the
raw user query before any further processing.  The function raises :class:`ValueError`
with a clear message when the input is not acceptable.

The validation rules are intentionally strict because the downstream LLM prompt
expects a well‑formed, non‑empty query string.

"""

from __future__ import annotations

import os
from typing import Any

from .exceptions import QueryValidationError

# Default maximum length for a query.  The value can be overridden via the
# environment variable ``MAX_QUERY_LENGTH`` to make the behaviour configurable
# without code changes.  The API uses the same limit for its request schema.
MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "256"))
_MAX_QUERY_LENGTH: int = MAX_QUERY_LENGTH


def _type_check(query: Any) -> None:
    """Internal helper to ensure *query* is a ``str``.

    Parameters
    ----------
    query:
        The value supplied by the caller.

    Raises
    ------
    QueryValidationError
        If *query* is not an instance of :class:`str`.
    """
    if not isinstance(query, str):
        raise QueryValidationError("Query must be a string")


def _content_check(query: str) -> None:
    """Internal helper to enforce non‑empty and length constraints.

    Parameters
    ----------
    query:
        Already verified to be a ``str``.

    Raises
    ------
    ValueError
        * If the stripped string is empty.
        * If the length exceeds the configured maximum.
    """
    stripped = query.strip()
    if not stripped:
        raise QueryValidationError("Query cannot be empty or whitespace only")
    if len(stripped) > _MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"Query exceeds maximum allowed length of {_MAX_QUERY_LENGTH} characters"
        )


def validate_query(query: Any) -> None:
    """Validate a raw user query.

    This function is the *public* validation entry point used by the
    reformulation pipeline.  It performs three checks:

    1. **Type check** – the value must be a ``str``.
    2. **Content check** – after stripping whitespace the string must be non‑empty.
    3. **Length check** – the stripped string may not be longer than the
       configured maximum (default 256 characters).

    If any check fails a :class:`ValueError` is raised with an explanatory
    message; otherwise the function returns ``None``.

    Parameters
    ----------
    query:
        The raw user input to validate.

    Raises
    ------
    ValueError
        If *query* fails any of the validation rules.
    """
    _type_check(query)
    _content_check(query)  # type: ignore[arg-type]  # after _type_check we know it's str
