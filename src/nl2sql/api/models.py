"""Request and response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """A question to answer."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question, in plain English.",
    )
    include_rows: bool = Field(
        default=True,
        description="Include the result rows. Set false when you only want the answer.",
    )
    max_rows: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Cap on rows returned in the response body.",
    )
    explain: bool = Field(
        default=False,
        description="Include the retrieval trace and per-attempt detail.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "How many accounts have more than 50000 in them?"},
                {"question": "Which region has the highest loan default rate?",
                 "explain": True},
            ]
        }
    }


class Usage(BaseModel):
    """Token and cost accounting for one request."""

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Timings(BaseModel):
    """Milliseconds spent in each stage. The first thing to look at when it is slow."""

    retrieval: float = 0.0
    generation: float = 0.0
    validation: float = 0.0
    execution: float = 0.0
    answer: float = 0.0
    total: float = 0.0


class AttemptInfo(BaseModel):
    """One pass through generate -> validate -> execute."""

    attempt: int
    outcome: str
    sql: str = ""
    error: str | None = None
    violations: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    """The answer, the query behind it, and how it was produced."""

    question: str
    answer: str
    sql: str

    success: bool
    refused: bool = False
    refusal_reason: str | None = None
    error: str | None = None

    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False

    tables_used: list[str] = Field(default_factory=list)
    repair_attempts: int = 0
    guardrail_blocked: bool = False
    warnings: list[str] = Field(default_factory=list)
    attempts: list[AttemptInfo] = Field(default_factory=list)

    usage: Usage = Field(default_factory=Usage)
    timings_ms: Timings = Field(default_factory=Timings)
    log_id: str = ""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How many accounts have more than 50000 in them?",
                    "answer": "1,627 accounts.",
                    "sql": "SELECT COUNT(*) AS account_count\nFROM account_summary\n"
                           "WHERE current_balance > 50000\nLIMIT 1000",
                    "success": True,
                    "columns": ["account_count"],
                    "rows": [[1627]],
                    "row_count": 1,
                    "tables_used": ["account_summary"],
                    "repair_attempts": 0,
                }
            ]
        }
    }


class ValidateRequest(BaseModel):
    """SQL to run past the guardrails."""

    sql: str = Field(..., min_length=1, max_length=20_000)

    model_config = {
        "json_schema_extra": {
            "examples": [{"sql": "SELECT 1; DROP TABLE loan"}]
        }
    }


class ValidateResponse(BaseModel):
    """What the guardrail layer decided."""

    accepted: bool
    sql: str = Field(description="The rewritten query that would actually run.")
    original_sql: str
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tables_referenced: list[str] = Field(default_factory=list)
    limit_applied: int | None = None
    limit_was_injected: bool = False


class TableSummary(BaseModel):
    name: str
    kind: str
    row_count: int
    column_count: int
    description: str = ""


class SchemaResponse(BaseModel):
    """The schema, either whole or as retrieved for a question."""

    tables: list[TableSummary]
    total_rows: int
    rendered: str = Field(description="The schema exactly as the model sees it.")
    retrieved_for: str | None = None


class HealthResponse(BaseModel):
    """Liveness and readiness."""

    status: str = Field(description="ok | degraded")
    version: str
    warehouse_ready: bool
    llm_provider: str
    llm_model: str
    api_key_present: bool
    tables: int
    stats: dict[str, Any] = Field(default_factory=dict)


class HistoryEntry(BaseModel):
    """One row of the audit log."""

    id: str
    created_at: str
    question: str
    sql: str | None = None
    answer: str | None = None
    success: bool
    refused: bool = False
    row_count: int | None = None
    repair_attempts: int = 0
    total_ms: float | None = None


class ErrorResponse(BaseModel):
    """A failure, in a shape clients can rely on."""

    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
