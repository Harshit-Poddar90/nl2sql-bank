"""Provider selection: turn a config value into a working client."""

from __future__ import annotations

from nl2sql.config import LLMProvider, Settings, get_settings
from nl2sql.exceptions import ConfigurationError
from nl2sql.llm.base import LLMClient
from nl2sql.llm.providers import (
    AnthropicClient,
    GeminiClient,
    OllamaClient,
    OpenAIClient,
)
from nl2sql.llm.stub import StubClient
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)

_REGISTRY: dict[LLMProvider, type[LLMClient]] = {
    LLMProvider.GEMINI: GeminiClient,
    LLMProvider.OPENAI: OpenAIClient,
    LLMProvider.ANTHROPIC: AnthropicClient,
    LLMProvider.OLLAMA: OllamaClient,
    LLMProvider.STUB: StubClient,
}


def create_llm_client(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    fallback_to_stub: bool = False,
) -> LLMClient:
    """Build the configured LLM client."""
    settings = settings or get_settings()
    chosen = provider or settings.llm_provider

    client_class = _REGISTRY.get(chosen)
    if client_class is None:  # pragma: no cover - unreachable via the enum
        raise ConfigurationError(
            f"Unknown LLM provider {chosen!r}. "
            f"Valid options: {', '.join(p.value for p in LLMProvider)}"
        )

    try:
        client = client_class(settings)
    except ConfigurationError:
        if not fallback_to_stub:
            raise
        log.warning(
            "llm_provider_unavailable_using_stub",
            extra={"requested": chosen.value},
        )
        return StubClient(settings)

    log.debug("llm_client_created", extra={"provider": chosen.value,
                                           "model": client.model})
    return client


def available_providers(settings: Settings | None = None) -> dict[str, bool]:
    """Which providers are usable right now, for ``/health`` and the CLI."""
    settings = settings or get_settings()
    status: dict[str, bool] = {}
    for provider in LLMProvider:
        try:
            _REGISTRY[provider](settings)
            status[provider.value] = True
        except Exception:
            status[provider.value] = False
    return status
