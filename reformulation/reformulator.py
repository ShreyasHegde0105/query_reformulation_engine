"""Query reformulation component.

This module provides a thin wrapper around the generic :class:`LLMClient`
implemented in ``reformulation.llm.client``.  It validates and normalizes the
raw user query, constructs a provider‑neutral prompt, calls the LLM, parses the
JSON response and returns a list of alternative search queries.

The implementation purposefully stays deterministic and offline‑friendly –
all network interaction is delegated to the injected ``LLMClient`` instance and
the tests mock that client.
"""

from __future__ import annotations

import json
from typing import List

from .validator import validate_query
from .normalizer import normalize_query
from .exceptions import ReformulationParseError, ReformulationError

# The LLM client lives in the ``reformulation.llm`` package.  Import lazily
# to avoid pulling in heavy dependencies unless actually used.
from .llm.client import LLMClient

__all__ = ["QueryReformulator"]


class QueryReformulator:
    """Generate alternative search‑query variants using an LLM.

    Parameters
    ----------
    llm_client:
        An instance of :class:`reformulation.llm.client.LLMClient`.  The client
        is expected to expose a ``generate(prompt: str) -> str`` method that
        returns the raw LLM response as a string.
    """

    _PROMPT_TEMPLATE = (
        "You are an AI assistant that generates semantically equivalent query reformulations "
        "useful for enterprise search. Produce concise, search-oriented query strings that "
        "preserve the original intent, entities, identifiers, acronyms, and technical terms. "
        "Do not answer the question or invent new facts. "
        "The original query should not be unnecessarily repeated. "
        "Return your response as a valid JSON object with the following structure:\n"
        '{{\n  "reformulations": [\n    "variant 1",\n    "variant 2"\n  ]\n}}\n'
        "The list must contain at most {max_variants} items. "
        'Use the following normalized query as the basis: "{normalized_query}"'
    )

    def __init__(self, llm_client: LLMClient):
        if not isinstance(llm_client, LLMClient):
            raise TypeError("llm_client must be an instance of LLMClient")
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_variants(self, query: str, max_variants: int = 5) -> List[str]:
        """Return a list of alternative search queries.

        The flow follows the contract defined in the specification:

        1. Validate the raw query via :func:`validate_query`.
        2. Normalize it via :func:`normalize_query`.
        3. Build a provider‑neutral prompt and call ``self.llm_client``.
        4. Parse the JSON response.
        5. Normalise each reformulated query, drop empties, remove exact
           duplicates (including the original normalized query), and limit the
           result to ``max_variants``.
        """
        # --- max_variants validation ------------------------------------------------
        if not isinstance(max_variants, int) or max_variants <= 0:
            raise ValueError("max_variants must be a positive integer")

        # --- validation & normalization --------------------------------------------
        validate_query(query)  # may raise ValueError
        normalized = normalize_query(query)

        # --- prompt construction ----------------------------------------------------
        prompt = self._PROMPT_TEMPLATE.format(
            max_variants=max_variants, normalized_query=normalized
        )

        # --- LLM call --------------------------------------------------------------
        llm_output = self.llm_client.generate(prompt)

        # --- JSON parsing ----------------------------------------------------------
        try:
            payload = json.loads(llm_output)
        except json.JSONDecodeError as exc:
            raise ReformulationParseError(
                f"LLM output is not valid JSON: {exc.msg}"
            ) from exc

        if not isinstance(payload, dict):
            raise ReformulationParseError("LLM JSON output must be an object")

        if "reformulations" not in payload:
            raise ReformulationParseError("Missing 'reformulations' key in LLM output")

        reformulations = payload["reformulations"]
        if not isinstance(reformulations, list):
            raise ReformulationParseError(
                "'reformulations' must be a list in LLM output"
            )

# --- item validation --------------------------------------------------------
        clean_variants: List[str] = []
        # First collect normalized candidates, excluding empties and the original query
        candidates: List[str] = []
        for item in reformulations:
            if not isinstance(item, str):
                raise ReformulationParseError(
                    "All items in 'reformulations' must be strings"
                )
            stripped = item.strip()
            if not stripped:
                continue
            norm_variant = normalize_query(stripped)
            if norm_variant == normalized:
                continue
            candidates.append(norm_variant)
        # Remove exact duplicates: use FIRST-OCCURRENCE-WINS
        seen = set()
        for var in candidates:
            if var not in seen:
                seen.add(var)
                clean_variants.append(var)
                if len(clean_variants) >= max_variants:
                    break

        return clean_variants

