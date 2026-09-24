"""The four hosted-model providers."""

from __future__ import annotations

from typing import Any

from nl2sql.config import Settings
from nl2sql.exceptions import ConfigurationError, LLMResponseError
from nl2sql.llm.base import HTTPLLMClient
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)


class GeminiClient(HTTPLLMClient):
    """Google Gemini via the Generative Language REST API."""

    provider = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.gemini_api_key:
            raise ConfigurationError(
                "No Gemini API key configured.",
                user_message=(
                    "Add your Gemini API key to the .env file:\n\n"
                    "    NL2SQL_GEMINI_API_KEY=your-key-here\n\n"
                    "Get a free key at https://aistudio.google.com/apikey\n"
                    "Or run without a key using: NL2SQL_LLM_PROVIDER=stub"
                ),
            )

    def _endpoint(self) -> str:
        return f"{self.BASE_URL}/models/{self.model}:generateContent"

    def _headers(self) -> dict[str, str]:
        # The key goes in a header, never in the URL query string -- a URL ends
        # up in proxy logs, browser history and error traces.
        return {
            "x-goog-api-key": self.settings.gemini_api_key,
            "Content-Type": "application/json",
        }

    def _payload(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                # Nudges the model towards the single best token rather than
                # sampling among plausible ones. For SQL there is usually one
                # right answer, so breadth is not a virtue.
                "topP": 0.95,
                "candidateCount": 1,
            },
        }

        if "2.5" in self.model or "2.0-flash-thinking" in self.model:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

        return payload

    def _parse(self, payload: dict[str, Any]) -> tuple[str, int, int, str]:
        usage = payload.get("usageMetadata", {})
        input_tokens = int(usage.get("promptTokenCount", 0))
        output_tokens = int(
            usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        )

        candidates = payload.get("candidates") or []
        if not candidates:
            # Usually a prompt-level safety block. Surface the stated reason
            # rather than a generic "empty response".
            feedback = payload.get("promptFeedback", {})
            raise LLMResponseError(
                f"Gemini returned no candidates. promptFeedback={feedback}",
                user_message="The model declined to answer this question.",
            )

        candidate = candidates[0]
        finish_reason = str(candidate.get("finishReason", "STOP")).lower()
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)

        return text, input_tokens, output_tokens, finish_reason


class OpenAIClient(HTTPLLMClient):
    """OpenAI chat completions."""

    provider = "openai"
    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ConfigurationError(
                "No OpenAI API key configured.",
                user_message=(
                    "Add your OpenAI API key to the .env file:\n\n"
                    "    NL2SQL_OPENAI_API_KEY=sk-...\n"
                ),
            )

    def _endpoint(self) -> str:
        return f"{self.BASE_URL}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

    def _parse(self, payload: dict[str, Any]) -> tuple[str, int, int, str]:
        usage = payload.get("usage", {})
        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        return (
            text,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            str(choice.get("finish_reason", "stop")),
        )


class AnthropicClient(HTTPLLMClient):
    """Anthropic Claude messages API."""

    provider = "anthropic"
    BASE_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.anthropic_api_key:
            raise ConfigurationError(
                "No Anthropic API key configured.",
                user_message=(
                    "Add your Anthropic API key to the .env file:\n\n"
                    "    NL2SQL_ANTHROPIC_API_KEY=sk-ant-...\n"
                ),
            )

    def _endpoint(self) -> str:
        return f"{self.BASE_URL}/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

    def _payload(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        # Anthropic takes the system prompt as a top-level field rather than as
        # a message with role 'system'.
        return {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _parse(self, payload: dict[str, Any]) -> tuple[str, int, int, str]:
        usage = payload.get("usage", {})
        blocks = payload.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return (
            text,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
            str(payload.get("stop_reason", "stop")),
        )


class OllamaClient(HTTPLLMClient):
    """A model running locally through Ollama."""

    provider = "ollama"

    def _endpoint(self) -> str:
        return f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _payload(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    def _parse(self, payload: dict[str, Any]) -> tuple[str, int, int, str]:
        text = (payload.get("message") or {}).get("content") or ""
        return (
            text,
            int(payload.get("prompt_eval_count", 0)),
            int(payload.get("eval_count", 0)),
            "stop" if payload.get("done") else "length",
        )
