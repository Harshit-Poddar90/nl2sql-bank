"""The exception hierarchy for nl2sql-bank."""

from __future__ import annotations


class NL2SQLError(Exception):
    """Base class for every error this project raises deliberately."""

    #: HTTP status the API layer should use if this escapes to a request handler.
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message
        self.details: dict[str, object] = details or {}



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ConfigurationError(NL2SQLError):
    """Something in .env or the environment is wrong or missing."""

    http_status = 500


# ---------------------------------------------------------------------------
# Data / warehouse
# ---------------------------------------------------------------------------
class DataError(NL2SQLError):
    """The warehouse is missing, empty, or structurally wrong."""

    http_status = 503


class DatasetDownloadError(DataError):
    """Could not fetch the source dataset from any configured mirror."""

    http_status = 503


class CatalogError(NL2SQLError):
    """The semantic catalog disagrees with the actual database."""

    http_status = 500


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
class LLMError(NL2SQLError):
    """The call to the language model failed."""

    http_status = 502


class LLMAuthError(LLMError):
    """The provider rejected our credentials (401/403)."""

    http_status = 401


class LLMRateLimitError(LLMError):
    """The provider is throttling us (429) or briefly unavailable (5xx)."""

    http_status = 429


class LLMResponseError(LLMError):
    """We got a response but it wasn't usable -- empty, truncated, or malformed."""

    http_status = 502


class SQLGenerationError(NL2SQLError):
    """Generation produced no SQL we could extract from the reply."""

    http_status = 422


# ---------------------------------------------------------------------------
# Safety and execution
# ---------------------------------------------------------------------------
class GuardrailViolation(NL2SQLError):  # noqa: N818 - see the note below
    """The generated SQL was rejected before it ever reached the database."""

    http_status = 400

    def __init__(
        self,
        message: str,
        *,
        violations: list[str] | None = None,
        sql: str | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            user_message=user_message
            or "The generated query was blocked by the safety layer.",
            details={"violations": violations or [], "sql": sql},
        )
        self.violations = violations or []
        self.sql = sql

    def feedback_for_model(self) -> str:
        """A terse, actionable description of what to fix, for the repair prompt."""
        if not self.violations:
            return self.message
        bullets = "\n".join(f"- {v}" for v in self.violations)
        return f"The query was rejected by the safety validator:\n{bullets}"


class QueryExecutionError(NL2SQLError):
    """The database refused to run the query, or failed while running it."""

    http_status = 400

    def __init__(
        self,
        message: str,
        *,
        sql: str | None = None,
        db_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            user_message="The query could not be run against the database.",
            details={"sql": sql, "db_message": db_message},
        )
        self.sql = sql
        self.db_message = db_message or message

    def feedback_for_model(self) -> str:
        return f"The database rejected the query with: {self.db_message}"


class QueryTimeoutError(QueryExecutionError):
    """The query exceeded its wall-clock budget and was interrupted."""

    http_status = 504

    def feedback_for_model(self) -> str:
        return (
            "The query was cancelled for taking too long. Rewrite it to be cheaper: "
            "add the missing join conditions, filter earlier, and avoid scanning "
            "the full transaction table without a WHERE clause."
        )
