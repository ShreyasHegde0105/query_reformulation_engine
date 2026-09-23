"""Unit tests for the LLM client implementation.

The tests use ``unittest.mock`` to replace the low‑level ``urllib.request``
calls, allowing us to verify payload construction, retry behaviour and error
mapping without performing real network I/O.
"""

import os
import json
import urllib
import unittest
from unittest import mock
from io import BytesIO

from reformulation.llm import (
    LLMClient,
    LLMConfig,
    LLMError,
    LLMConfigError,
    LLMPromptError,
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
)

# Helper to build a minimal valid configuration
def _make_config(**overrides):
    base = {
        "provider": "test",
        "model": "gpt-test",
        "api_key": "testkey",
        "base_url": "http://example.com/v1/chat",
        "temperature": 0.0,
        "timeout_seconds": 5,
        "max_retries": 2,
    }
    base.update(overrides)
    return LLMConfig(**base)

# Simple mock response object compatible with ``urllib`` context manager
class _MockHTTPResponse:
    def __init__(self, data: bytes, code: int = 200):
        self._data = data
        self._code = code
    def read(self):
        return self._data
    def getcode(self):
        return self._code
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

class TestLLMClient(unittest.TestCase):
    def setUp(self):
        # Use a no‑op sleep to keep tests fast
        self.client = LLMClient(_make_config(), sleep_func=lambda _: None)

    # ---------- Configuration ----------
    def test_valid_configuration_direct(self):
        cfg = _make_config()
        self.assertIsInstance(cfg, LLMConfig)
        self.assertEqual(cfg.model, "gpt-test")

    def test_configuration_from_env(self):
        env = {
            "LLM_PROVIDER": "envprov",
            "LLM_MODEL": "envmodel",
            "LLM_API_KEY": "envkey",
            "LLM_BASE_URL": "http://env.com/api",
            "LLM_TEMPERATURE": "0.7",
            "LLM_TIMEOUT_SECONDS": "10",
            "LLM_MAX_RETRIES": "3",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig.from_env()
            self.assertEqual(cfg.provider, "envprov")
            self.assertAlmostEqual(cfg.temperature, 0.7)
            self.assertEqual(cfg.timeout_seconds, 10)
            self.assertEqual(cfg.max_retries, 3)

    def test_invalid_configuration_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigError):
                LLMConfig.from_env()

    # ---------- Prompt validation ----------
    def test_generate_non_string_prompt(self):
        with self.assertRaises(LLMPromptError):
            self.client.generate(123)  # type: ignore

    def test_generate_empty_prompt(self):
        with self.assertRaises(LLMPromptError):
            self.client.generate("")

    def test_generate_whitespace_prompt(self):
        with self.assertRaises(LLMPromptError):
            self.client.generate("   \t\n  ")

    # ---------- Successful generation ----------
    @mock.patch("urllib.request.urlopen")
    def test_successful_generate(self, mock_urlopen):
        expected_text = "generated response"
        payload = {"choices": [{"message": {"content": expected_text}}]}
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(payload).encode())
        result = self.client.generate("hello")
        self.assertEqual(result, expected_text)
        # Verify request payload structure
        args, _ = mock_urlopen.call_args
        request = args[0]
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertIn("Bearer", request.get_header("Authorization"))
        body = json.loads(request.data.decode())
        self.assertEqual(body["model"], self.client.config.model)
        self.assertEqual(body["messages"][0]["content"], "hello")
        self.assertEqual(body["temperature"], self.client.config.temperature)

    # ---------- Authentication error ----------
    @mock.patch("urllib.request.urlopen")
    def test_auth_error_401(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://x", code=401, msg="unauth", hdrs=None, fp=None
        )
        with self.assertRaises(LLMAuthError):
            self.client.generate("test")
        # No retries should be performed
        self.assertEqual(mock_urlopen.call_count, 1)

    # ---------- Rate limiting with retry ----------
    @mock.patch("urllib.request.urlopen")
    def test_rate_limit_retry_success(self, mock_urlopen):
        # First call 429, second call success
        err = urllib.error.HTTPError(url="http://x", code=429, msg="rate", hdrs=None, fp=None)
        success_payload = {"choices": [{"message": {"content": "ok"}}]}
        mock_urlopen.side_effect = [err, _MockHTTPResponse(json.dumps(success_payload).encode())]
        result = self.client.generate("retry")
        self.assertEqual(result, "ok")
        self.assertEqual(mock_urlopen.call_count, 2)

    @mock.patch("urllib.request.urlopen")
    def test_rate_limit_exhausted(self, mock_urlopen):
        err = urllib.error.HTTPError(url="http://x", code=429, msg="rate", hdrs=None, fp=None)
        mock_urlopen.side_effect = [err, err, err]
        with self.assertRaises(LLMRateLimitError):
            self.client.generate("fail")
        self.assertEqual(mock_urlopen.call_count, 3)  # max_retries=2 => 3 attempts total

    # ---------- Server error (5xx) retry ----------
    @mock.patch("urllib.request.urlopen")
    def test_server_error_retry_success(self, mock_urlopen):
        err = urllib.error.HTTPError(url="http://x", code=502, msg="bad", hdrs=None, fp=None)
        success_payload = {"choices": [{"message": {"content": "ok"}}]}
        mock_urlopen.side_effect = [err, _MockHTTPResponse(json.dumps(success_payload).encode())]
        result = self.client.generate("retry5xx")
        self.assertEqual(result, "ok")
        self.assertEqual(mock_urlopen.call_count, 2)

    @mock.patch("urllib.request.urlopen")
    def test_server_error_exhausted(self, mock_urlopen):
        err = urllib.error.HTTPError(url="http://x", code=503, msg="bad", hdrs=None, fp=None)
        mock_urlopen.side_effect = [err, err, err]
        with self.assertRaises(LLMResponseError):
            self.client.generate("fail5xx")
        self.assertEqual(mock_urlopen.call_count, 3)

    # ---------- Timeout handling ----------
    @mock.patch("urllib.request.urlopen")
    def test_timeout_retry_success(self, mock_urlopen):
        timeout_err = urllib.error.URLError(reason=TimeoutError())
        success_payload = {"choices": [{"message": {"content": "ok"}}]}
        mock_urlopen.side_effect = [timeout_err, _MockHTTPResponse(json.dumps(success_payload).encode())]
        result = self.client.generate("timeout")
        self.assertEqual(result, "ok")
        self.assertEqual(mock_urlopen.call_count, 2)

    @mock.patch("urllib.request.urlopen")
    def test_timeout_exhausted(self, mock_urlopen):
        timeout_err = urllib.error.URLError(reason=TimeoutError())
        mock_urlopen.side_effect = [timeout_err, timeout_err, timeout_err]
        with self.assertRaises(LLMTimeoutError):
            self.client.generate("fail_timeout")
        self.assertEqual(mock_urlopen.call_count, 3)

    # ---------- Malformed response ----------
    @mock.patch("urllib.request.urlopen")
    def test_malformed_response(self, mock_urlopen):
        bad_payload = {"unexpected": []}
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(bad_payload).encode())
        with self.assertRaises(LLMResponseError):
            self.client.generate("bad")

    # ---------- Connection failure (non‑timeout) ----------
    @mock.patch("urllib.request.urlopen")
    def test_connection_failure_exhausted(self, mock_urlopen):
        conn_err = urllib.error.URLError(reason=RuntimeError("network down"))
        mock_urlopen.side_effect = [conn_err, conn_err, conn_err]
        with self.assertRaises(LLMError):
            self.client.generate("conn")
        self.assertEqual(mock_urlopen.call_count, 3)

    # ---------- Max retries configuration respected
    def test_custom_max_retries(self):
        cfg = _make_config(max_retries=0)
        client = LLMClient(cfg, sleep_func=lambda _: None)
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            err = urllib.error.HTTPError(url="http://x", code=429, msg="rate", hdrs=None, fp=None)
            mock_urlopen.side_effect = err
            with self.assertRaises(LLMRateLimitError):
                client.generate("test")
            self.assertEqual(mock_urlopen.call_count, 1)

if __name__ == "__main__":
    unittest.main()
