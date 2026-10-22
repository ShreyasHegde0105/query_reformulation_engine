"""Provider‑agnostic LLM HTTP client.

The client uses only the Python standard library to keep dependencies minimal
and to simplify mocking in the test suite.  It follows the specification for
prompt validation, request construction, authentication, retries, timeout
handling and response validation.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

from .config import LLMConfig
from .exceptions import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMPromptError,
    LLMError,
)


class LLMClient:
    """Simple HTTP client for LLM providers.

    Parameters
    ----------
    config: LLMConfig
        Configuration containing provider details, auth information and retry
        settings.
    sleep_func: Callable[[float], None] | None, optional
        Function used for back‑off sleeps.  Defaults to :func:`time.sleep`.  Tests
        can inject a dummy function to avoid real delays.
    """

    def __init__(self, config: LLMConfig, sleep_func: Optional[Callable[[float], None]] = None):
        self.config = config
        self._sleep = sleep_func or time.sleep

    # ---------------------------------------------------------------------
    # Prompt validation
    # ---------------------------------------------------------------------
    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        if not isinstance(prompt, str):
            raise LLMPromptError("Prompt must be a string")
        if not prompt.strip():
            raise LLMPromptError("Prompt cannot be empty or whitespace only")

    # ---------------------------------------------------------------------
    # Payload construction
    # ---------------------------------------------------------------------
    def _build_payload(self, prompt: str) -> bytes:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        return json.dumps(payload).encode("utf-8")

    _MAX_RETRY_AFTER_SECONDS = 30.0

    @classmethod
    def _retry_after_seconds(cls, http_err: urllib.error.HTTPError) -> float:
        """Seconds requested by a ``Retry-After`` header (capped), or 0 if absent/unparseable."""
        headers = getattr(http_err, "headers", None)
        value = headers.get("Retry-After") if headers is not None else None
        try:
            return min(max(float(value), 0.0), cls._MAX_RETRY_AFTER_SECONDS)
        except (TypeError, ValueError):
            return 0.0

    # ---------------------------------------------------------------------
    # Low‑level POST with retry handling
    # ---------------------------------------------------------------------
    def _post(self, data: bytes) -> dict:
        url = self.config.base_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            # Some providers' firewalls (e.g. Groq's) block urllib's default User-Agent with HTTP 403.
            "User-Agent": "query-reformulation-engine/1.0",
        }
        request = urllib.request.Request(url, data=data, method="POST")
        for header_name, header_value in headers.items():
            request.add_header(header_name, header_value)

        attempts = 0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except urllib.error.HTTPError as http_err:
                status = http_err.code
                # 401 – authentication failure (do not retry)
                if status == 401:
                    raise LLMAuthError("Authentication failed (HTTP 401)") from http_err
                # 429 – rate limiting (retry according to max_retries)
                if status == 429:
                    attempts += 1
                    if attempts > self.config.max_retries:
                        # Surface the provider's explanation (e.g. which quota was hit)
                        try:
                            detail = http_err.read().decode("utf-8", errors="replace")[:1000]
                        except Exception:
                            detail = ""
                        message = "Rate limit exceeded after retries (HTTP 429)"
                        if detail:
                            message += f": {detail}"
                        raise LLMRateLimitError(message) from http_err
                    backoff = max((2 ** (attempts - 1)) * 0.1, self._retry_after_seconds(http_err))
                    self._sleep(backoff)
                    continue
                # 5xx – server errors (retry)
                if 500 <= status < 600:
                    attempts += 1
                    if attempts > self.config.max_retries:
                        raise LLMResponseError(
                            f"Server error {status} after {self.config.max_retries} retries"
                        ) from http_err
                    backoff = (2 ** (attempts - 1)) * 0.1
                    self._sleep(backoff)
                    continue
                # Other HTTP errors are treated as response errors
                raise LLMResponseError(f"Unexpected HTTP status {status}") from http_err
            except urllib.error.URLError as url_err:
                # Timeout handling – ``reason`` may be a ``socket.timeout``
                if isinstance(url_err.reason, TimeoutError):
                    attempts += 1
                    if attempts > self.config.max_retries:
                        raise LLMTimeoutError("Request timed out after retries") from url_err
                    backoff = (2 ** (attempts - 1)) * 0.1
                    self._sleep(backoff)
                    continue
                # Generic connection failure – retry similarly
                attempts += 1
                if attempts > self.config.max_retries:
                    raise LLMError(f"Connection failed after {self.config.max_retries} retries") from url_err
                backoff = (2 ** (attempts - 1)) * 0.1
                self._sleep(backoff)
                continue
            except TimeoutError as timeout_err:
                # A timeout while reading the response is raised directly, not wrapped in URLError.
                attempts += 1
                if attempts > self.config.max_retries:
                    raise LLMTimeoutError("Request timed out after retries") from timeout_err
                backoff = (2 ** (attempts - 1)) * 0.1
                self._sleep(backoff)
                continue
            except Exception as exc:
                raise LLMError(f"Unexpected error during LLM request ({type(exc).__name__})") from exc

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """Generate a completion for *prompt*.

        The method validates the prompt, builds the request payload, sends the
        HTTP request (with retries) and extracts the generated text from the
        provider's response.  All error conditions map to the custom exception
        hierarchy.
        """
        self._validate_prompt(prompt)
        payload = self._build_payload(prompt)
        response = self._post(payload)

        # Expected shape: {"choices": [{"message": {"content": "..."}}]}
        try:
            choices = response["choices"]
            if not isinstance(choices, list) or not choices:
                raise KeyError
            message = choices[0]["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError
            return content
        except Exception as exc:
            raise LLMResponseError("Malformed response from LLM provider") from exc

    # ---------------------------------------------------------------------
    # Test‑utility hook – replace the request method with a mock
    # ---------------------------------------------------------------------
    def _set_request_func(self, func: Callable[[bytes], dict]):
        """Replace the internal ``_post`` method with *func*.

        ``func`` receives the request payload (bytes) and must return a parsed
        JSON ``dict``.  This is primarily for unit tests that want to avoid real
        network I/O while still exercising the client logic.
        """
        self._post = func
