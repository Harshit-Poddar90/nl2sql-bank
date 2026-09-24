"""The LLM interface, and the retry logic every provider shares."""

from __future__ import annotations

import contextlib
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from nl2sql.config import Settings
from nl2sql.exceptions import LLMAuthError, LLMError, LLMRateLimitError, LLMResponseError
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class LLMResponse:
    """One completion, plus everything needed to account for it."""

    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"


class LLMClient(ABC):
    """What the rest of the system is allowed to assume about a model."""

    provider: str = "unknown"

    #: Whether this client can write natural-language prose, as opposed to only
    #: SQL. Every real model can; the rule-based stub cannot -- ask it to
    #: summarise a result set and it returns a SELECT statement, because
    #: emitting SQL is the only thing it does.
    #:
    #: The answer synthesizer checks this before delegating. Without it, running
    #: offline produces answers that are literally a block of SQL where a
    #: sentence should be.
    supports_prose: bool = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.resolved_model

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion."""


class HTTPLLMClient(LLMClient):
    """Shared HTTP plumbing: one request shape, one retry policy, one error map."""

    #: Status codes worth trying again. 408 request timeout, 409 conflict,
    #: 429 rate limit, and the 5xx range: all transient by definition.
    RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._client = httpx.Client(timeout=settings.llm_timeout_seconds)

    # -- subclass hooks -----------------------------------------------------
    @abstractmethod
    def _endpoint(self) -> str:
        """Full URL to POST to."""

    @abstractmethod
    def _headers(self) -> dict[str, str]:
        """Request headers, including authentication."""

    @abstractmethod
    def _payload(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        """The JSON request body."""

    @abstractmethod
    def _parse(self, payload: dict[str, Any]) -> tuple[str, int, int, str]:
        """Extract ``(text, input_tokens, output_tokens, finish_reason)``."""

    # -- the shared implementation ------------------------------------------
    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        temperature = (
            self.settings.llm_temperature if temperature is None else temperature
        )
        max_tokens = max_tokens or self.settings.llm_max_output_tokens

        body = self._payload(system, user, temperature, max_tokens)
        last_error: Exception | None = None

        # attempts = 1 initial try + llm_max_retries retries.
        for attempt in range(1, self.settings.llm_max_retries + 2):
            try:
                response = self._client.post(
                    self._endpoint(), headers=self._headers(), json=body
                )

                if response.status_code in self.RETRYABLE_STATUS:
                    last_error = LLMRateLimitError(
                        f"{self.provider} returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    self._sleep_before_retry(attempt, response)
                    continue

                if response.status_code in (401, 403):
                    raise LLMAuthError(
                        f"{self.provider} rejected the API key "
                        f"({response.status_code}).",
                        user_message=(
                            f"Your {self.provider} API key is missing or invalid. "
                            f"Set NL2SQL_{self.provider.upper()}_API_KEY in your .env file."
                        ),
                    )

                if response.status_code >= 400:
                    # A 4xx that is not auth and not rate-limiting is our bug --
                    # a malformed request. Retrying identical bad input would
                    # just waste time and quota.
                    raise LLMError(
                        f"{self.provider} returned {response.status_code}: "
                        f"{response.text[:500]}",
                        user_message=f"The {self.provider} API rejected the request.",
                    )

                text, input_tokens, output_tokens, finish_reason = self._parse(
                    response.json()
                )

                if not text.strip():
                    # Empty completions do happen -- a safety filter, or a
                    # thinking-enabled model that spent its whole budget before
                    # emitting anything. Worth one more try.
                    last_error = LLMResponseError(
                        f"{self.provider} returned an empty completion "
                        f"(finish_reason={finish_reason})."
                    )
                    self._sleep_before_retry(attempt, None)
                    continue

                return LLMResponse(
                    text=text,
                    provider=self.provider,
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason=finish_reason,
                )

            except (LLMAuthError, LLMError) as exc:
                if isinstance(exc, LLMRateLimitError):
                    last_error = exc
                    self._sleep_before_retry(attempt, None)
                    continue
                raise

            except httpx.TimeoutException as exc:
                last_error = LLMError(f"{self.provider} timed out: {exc}")
                self._sleep_before_retry(attempt, None)

            except httpx.HTTPError as exc:
                last_error = LLMError(f"{self.provider} transport error: {exc}")
                self._sleep_before_retry(attempt, None)

            except (KeyError, IndexError, ValueError) as exc:
                raise LLMResponseError(
                    f"Could not parse the {self.provider} response: {exc}",
                    user_message="The language model returned an unexpected response shape.",
                ) from exc

        raise LLMError(
            f"{self.provider} failed after {self.settings.llm_max_retries + 1} attempts: "
            f"{last_error}",
            user_message=(
                f"Could not reach the {self.provider} API. Check your network "
                f"connection and API key."
            ),
        ) from last_error

    def _sleep_before_retry(self, attempt: int, response: httpx.Response | None) -> None:
        """Exponential backoff with jitter, respecting Retry-After when offered."""
        if attempt > self.settings.llm_max_retries:
            return

        delay = min(2.0 ** (attempt - 1), 30.0)

        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                # Retry-After may be an HTTP date rather than seconds. If it
                # will not parse as a number, our own backoff is fine.
                with contextlib.suppress(ValueError):
                    delay = max(delay, float(retry_after))

        delay += random.uniform(0, 0.5 * delay)
        log.warning(
            "llm_retry",
            extra={"provider": self.provider, "attempt": attempt,
                   "sleep_seconds": round(delay, 2)},
        )
        time.sleep(delay)

