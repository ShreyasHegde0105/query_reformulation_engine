"""LLM client configuration.

Provides :class:`LLMConfig` which loads configuration from environment variables
or can be instantiated directly. Validation ensures all required fields are
present and have sensible types.

Two ways to configure the LLM in ``.env``:

* **Provider keys (recommended).** Paste keys only, e.g. ``GEMINI_API_KEY``,
  ``GEMINI_API_KEY_2``, ``GROQ_API_KEY``. Endpoints and models come from
  :data:`PROVIDER_PRESETS`; :func:`provider_configs_from_env` returns one config
  per key, and :class:`~reformulation.llm.fallback.FallbackLLMClient` switches to
  the next key when one runs out of quota.
* **One explicit endpoint.** ``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_API_KEY`` and
  ``LLM_BASE_URL`` for any OpenAI-compatible ``/chat/completions`` endpoint.
"""

import os
from dataclasses import dataclass, replace
from typing import List, Optional

from .exceptions import LLMConfigError

@dataclass(frozen=True)
class LLMConfig:
    """Configuration for :class:`~reformulation.llm.client.LLMClient`.

    Parameters are loaded from environment variables if not provided
    explicitly. Validation occurs at instantiation and raises
    :class:`LLMConfigError` on missing or malformed values.

    ``temperature=None`` leaves temperature out of the request, for models that
    only accept their default (e.g. OpenAI's GPT-5 family).
    """

    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: Optional[float] = 0.0
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
    def explicit_from_env(cls) -> Optional["LLMConfig"]:
        """The ``LLM_PROVIDER`` / ``LLM_MODEL`` / ``LLM_API_KEY`` / ``LLM_BASE_URL`` config, or None if not set."""
        try:
            provider = os.getenv("LLM_PROVIDER")
            model = os.getenv("LLM_MODEL")
            api_key = os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL")
            if not all([provider, model, api_key, base_url]):
                return None
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

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create a single configuration instance from environment variables.

        Uses the explicit ``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_API_KEY`` and
        ``LLM_BASE_URL`` variables when all are set; otherwise the first
        provider key from :func:`provider_configs_from_env`. Optional variables
        with defaults are ``LLM_TEMPERATURE``, ``LLM_TIMEOUT_SECONDS`` and
        ``LLM_MAX_RETRIES``.

        Evaluation scripts use this (one fixed model per benchmark run); the
        live demo and API use :class:`~reformulation.llm.fallback.FallbackLLMClient`.
        """
        explicit = cls.explicit_from_env()
        if explicit is not None:
            return explicit
        configs = provider_configs_from_env()
        if not configs:
            raise LLMConfigError(
                "Missing required LLM configuration environment variables "
                "(set a provider key such as GEMINI_API_KEY, or LLM_PROVIDER/LLM_MODEL/LLM_API_KEY/LLM_BASE_URL)"
            )
        max_retries = cls._parse_int(os.getenv("LLM_MAX_RETRIES", "2"), "LLM_MAX_RETRIES")
        return replace(configs[0], max_retries=max_retries)


@dataclass(frozen=True)
class ProviderPreset:
    """Endpoint and default model for one OpenAI-compatible provider."""

    name: str
    env_prefix: str
    base_url: str
    default_model: str
    send_temperature: bool = True


PROVIDER_PRESETS = {p.name: p for p in (
    ProviderPreset("gemini", "GEMINI",
                   "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                   "gemini-3.5-flash-lite"),
    ProviderPreset("groq", "GROQ", "https://api.groq.com/openai/v1/chat/completions",
                   "openai/gpt-oss-120b"),
    ProviderPreset("openrouter", "OPENROUTER", "https://openrouter.ai/api/v1/chat/completions",
                   "openrouter/free"),
    ProviderPreset("zai", "ZAI", "https://api.z.ai/api/paas/v4/chat/completions",
                   "glm-4.5-flash", send_temperature=False),
    ProviderPreset("openai", "OPENAI", "https://api.openai.com/v1/chat/completions",
                   "gpt-5-nano", send_temperature=False),
)}
# Free tiers first, paid last.
DEFAULT_PROVIDER_ORDER = tuple(PROVIDER_PRESETS)
MAX_KEYS_PER_PROVIDER = 9


def provider_configs_from_env() -> List[LLMConfig]:
    """One :class:`LLMConfig` per provider key found in the environment, in fail-over order.

    For each provider, ``<PREFIX>_API_KEY`` and ``<PREFIX>_API_KEY_2`` ...
    ``_9`` are read (e.g. ``GEMINI_API_KEY_2``); blank keys are skipped.
    ``<PREFIX>_MODEL`` overrides the default model. ``LLM_PROVIDER_ORDER``
    (comma-separated names) overrides :data:`DEFAULT_PROVIDER_ORDER`, and
    ``LLM_KEY_RETRIES`` (default 0) sets retries per key before moving on.
    """
    order_env = os.getenv("LLM_PROVIDER_ORDER", "")
    order = [n.strip().lower() for n in order_env.split(",") if n.strip()] or list(DEFAULT_PROVIDER_ORDER)
    unknown = [n for n in order if n not in PROVIDER_PRESETS]
    if unknown:
        raise LLMConfigError(
            f"Unknown provider(s) in LLM_PROVIDER_ORDER: {', '.join(unknown)} "
            f"(known: {', '.join(PROVIDER_PRESETS)})"
        )
    temperature = LLMConfig._parse_float(os.getenv("LLM_TEMPERATURE", "0.0"), "LLM_TEMPERATURE")
    timeout_seconds = LLMConfig._parse_int(os.getenv("LLM_TIMEOUT_SECONDS", "30"), "LLM_TIMEOUT_SECONDS")
    key_retries = LLMConfig._parse_int(os.getenv("LLM_KEY_RETRIES", "0"), "LLM_KEY_RETRIES")

    configs: List[LLMConfig] = []
    for name in order:
        preset = PROVIDER_PRESETS[name]
        model = os.getenv(f"{preset.env_prefix}_MODEL", "").strip() or preset.default_model
        for i in range(1, MAX_KEYS_PER_PROVIDER + 1):
            var = f"{preset.env_prefix}_API_KEY" if i == 1 else f"{preset.env_prefix}_API_KEY_{i}"
            key = os.getenv(var, "").strip()
            if not key:
                continue
            configs.append(LLMConfig(
                provider=name,
                model=model,
                api_key=key,
                base_url=preset.base_url,
                temperature=temperature if preset.send_temperature else None,
                timeout_seconds=timeout_seconds,
                max_retries=key_retries,
            ))
    return configs
