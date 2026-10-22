import json
import os
import unittest
from unittest import mock

from reformulation.llm import (
    FallbackLLMClient,
    LLMAuthError,
    LLMClient,
    LLMConfig,
    LLMConfigError,
    LLMPromptError,
    LLMRateLimitError,
    LLMResponseError,
)
from reformulation.llm.config import PROVIDER_PRESETS, provider_configs_from_env


def _client(provider="test", reply=None, error=None):
    client = LLMClient(LLMConfig(provider=provider, model="m", api_key=f"{provider}-key", base_url="http://x"))
    calls = []

    def post(_data):
        calls.append(1)
        if error is not None:
            raise error
        return {"choices": [{"message": {"content": reply}}]}

    client._set_request_func(post)
    client.calls = calls
    return client


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TestProviderConfigsFromEnv(unittest.TestCase):
    def test_reads_numbered_keys_in_default_order(self):
        env = {"GROQ_API_KEY": "g", "GEMINI_API_KEY": "a", "GEMINI_API_KEY_2": "b", "OPENAI_API_KEY": "o"}
        with mock.patch.dict(os.environ, env, clear=True):
            configs = provider_configs_from_env()
        self.assertEqual([(c.provider, c.api_key) for c in configs],
                         [("gemini", "a"), ("gemini", "b"), ("groq", "g"), ("openai", "o")])
        self.assertEqual(configs[0].base_url, PROVIDER_PRESETS["gemini"].base_url)
        self.assertEqual(configs[0].max_retries, 0)
        self.assertIsNone(configs[-1].temperature)   # OpenAI preset omits temperature

    def test_blank_keys_skipped_and_model_override(self):
        env = {"GEMINI_API_KEY": "  ", "ZAI_API_KEY": "z", "ZAI_MODEL": "glm-x"}
        with mock.patch.dict(os.environ, env, clear=True):
            configs = provider_configs_from_env()
        self.assertEqual([(c.provider, c.model) for c in configs], [("zai", "glm-x")])

    def test_custom_order(self):
        env = {"GEMINI_API_KEY": "a", "GROQ_API_KEY": "g", "LLM_PROVIDER_ORDER": "groq, gemini"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual([c.provider for c in provider_configs_from_env()], ["groq", "gemini"])

    def test_unknown_provider_in_order(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "gemini,nope"}, clear=True):
            with self.assertRaises(LLMConfigError):
                provider_configs_from_env()

    def test_single_config_falls_back_to_first_provider_key(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "g"}, clear=True):
            cfg = LLMConfig.from_env()
        self.assertEqual((cfg.provider, cfg.model), ("groq", PROVIDER_PRESETS["groq"].default_model))
        self.assertEqual(cfg.max_retries, 2)


class TestFallbackFromEnv(unittest.TestCase):
    def test_labels_and_explicit_endpoint_first(self):
        env = {"LLM_PROVIDER": "custom", "LLM_MODEL": "m", "LLM_API_KEY": "k", "LLM_BASE_URL": "http://x",
               "GEMINI_API_KEY": "a", "GEMINI_API_KEY_2": "b", "GROQ_API_KEY": "g"}
        with mock.patch.dict(os.environ, env, clear=True):
            client = FallbackLLMClient.from_env()
        self.assertEqual(client.labels, ["custom", "gemini key 1", "gemini key 2", "groq"])

    def test_explicit_key_not_duplicated(self):
        env = {"LLM_PROVIDER": "google", "LLM_MODEL": "m", "LLM_API_KEY": "a", "LLM_BASE_URL": "http://x",
               "GEMINI_API_KEY": "a"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(FallbackLLMClient.from_env().labels, ["google"])

    def test_no_keys(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigError):
                FallbackLLMClient.from_env()


class TestFallbackGenerate(unittest.TestCase):
    def test_first_key_answers(self):
        a, b = _client("a", reply="one"), _client("b", reply="two")
        self.assertEqual(FallbackLLMClient([a, b]).generate("p"), "one")
        self.assertEqual(len(b.calls), 0)

    def test_switches_on_quota_and_cools_down(self):
        clock = _Clock()
        a = _client("a", error=LLMRateLimitError("429"))
        b = _client("b", reply="two")
        client = FallbackLLMClient([a, b], clock=clock)
        with self.assertLogs("reformulation.llm.fallback", level="WARNING") as logs:
            self.assertEqual(client.generate("p"), "two")
        self.assertIn("switching to b", "\n".join(logs.output))
        self.assertEqual(client.active, "b")

        client.generate("p")                      # a is cooling down: not retried
        self.assertEqual(len(a.calls), 1)
        clock.now += client.cooldown_seconds + 1
        client.generate("p")                      # cooldown over: a is tried again
        self.assertEqual(len(a.calls), 2)

    def test_rejected_key_dropped_for_session(self):
        clock = _Clock()
        a = _client("a", error=LLMAuthError("401"))
        b = _client("b", reply="two")
        client = FallbackLLMClient([a, b], clock=clock)
        client.generate("p")
        clock.now += 10 * client.cooldown_seconds
        client.generate("p")
        self.assertEqual(len(a.calls), 1)

    def test_all_fail_raises_last_error(self):
        a = _client("a", error=LLMRateLimitError("429"))
        b = _client("b", error=LLMResponseError("500"))
        with self.assertLogs("reformulation.llm.fallback", level="WARNING"):
            with self.assertRaises(LLMResponseError):
                FallbackLLMClient([a, b]).generate("p")

    def test_all_cooling_down_still_tries(self):
        a = _client("a", error=LLMRateLimitError("429"))
        client = FallbackLLMClient([a])
        with self.assertLogs("reformulation.llm.fallback", level="WARNING"):
            for _ in range(2):
                with self.assertRaises(LLMRateLimitError):
                    client.generate("p")
        self.assertEqual(len(a.calls), 2)

    def test_bad_prompt_not_sent(self):
        a = _client("a", reply="x")
        with self.assertRaises(LLMPromptError):
            FallbackLLMClient([a]).generate("  ")
        self.assertEqual(len(a.calls), 0)


class TestTemperatureOmitted(unittest.TestCase):
    def test_payload_without_temperature(self):
        client = LLMClient(LLMConfig(provider="openai", model="m", api_key="k", base_url="http://x", temperature=None))
        body = json.loads(client._build_payload("hi"))
        self.assertNotIn("temperature", body)


if __name__ == "__main__":
    unittest.main()
