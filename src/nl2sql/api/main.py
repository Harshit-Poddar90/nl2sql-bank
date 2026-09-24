"""The HTTP API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from nl2sql import __version__
from nl2sql.api.models import (
    AskRequest,
    AskResponse,
    AttemptInfo,
    ErrorResponse,
    HealthResponse,
    HistoryEntry,
    SchemaResponse,
    TableSummary,
    Timings,
    Usage,
    ValidateRequest,
    ValidateResponse,
)
from nl2sql.config import get_settings
from nl2sql.data.etl import warehouse_exists
from nl2sql.exceptions import NL2SQLError
from nl2sql.logging_setup import configure_logging, get_logger
from nl2sql.observability.query_log import QueryLog
from nl2sql.pipeline import Pipeline

log = get_logger(__name__)

#: Built once in the lifespan handler and shared by every request. The pipeline
#: holds no per-request state, so this is safe across threads.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Build the pipeline before serving, tear it down after."""
    configure_logging()
    settings = get_settings()

    if not warehouse_exists(settings):
        # Start anyway, so /health can report the problem rather than the
        # container crash-looping with the reason buried in the logs.
        log.error("warehouse_missing", extra={"hint": "run: nl2sql data build"})
        _state["pipeline"] = None
    else:
        try:
            # fallback_to_stub keeps the service usable with no API key: it
            # serves rule-based answers and says so via /health, instead of
            # returning 500 to every caller.
            _state["pipeline"] = Pipeline.build(settings, fallback_to_stub=True)
        except Exception as exc:
            log.exception("pipeline_build_failed", extra={"error": str(exc)})
            _state["pipeline"] = None

    _state["query_log"] = QueryLog(settings=settings)
    log.info("api_ready", extra={"version": __version__,
                                 "ready": _state["pipeline"] is not None})
    yield

    pipeline = _state.get("pipeline")
    if pipeline is not None:
        pipeline.close()
    log.info("api_shutdown")


app = FastAPI(
    title="nl2sql-bank",
    version=__version__,
    lifespan=lifespan,
    description=(
        "Ask a 1.06M-row banking warehouse questions in plain English. "
        "Every answer comes with the SQL that produced it, and every query is "
        "checked by an AST-level safety validator before it runs."
    ),
)


@app.exception_handler(NL2SQLError)
async def handle_nl2sql_error(_request: Request, exc: NL2SQLError) -> JSONResponse:
    """Turn our exceptions into the status and message they declare."""
    log.warning("api_error", extra={"error_type": type(exc).__name__,
                                    "message": exc.message[:200]})
    return JSONResponse(
        status_code=exc.http_status,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=exc.user_message,
            details=exc.details,
        ).model_dump(),
    )


def _require_pipeline() -> Pipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="The warehouse is not ready. Run `nl2sql data build` and restart.",
        )
    return pipeline


# ===========================================================================
# Endpoints
# ===========================================================================

@app.post("/ask", response_model=AskResponse, tags=["query"])
def ask(request: AskRequest) -> AskResponse:
    """Answer a question in plain English."""
    pipeline = _require_pipeline()
    result = pipeline.ask(request.question)

    rows: list[list[Any]] = []
    if request.include_rows:
        rows = [list(row) for row in result.rows[: request.max_rows]]

    return AskResponse(
        question=result.question,
        answer=result.answer,
        sql=result.sql,
        success=result.success,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        error=result.error,
        columns=result.columns if request.include_rows else [],
        rows=rows,
        row_count=result.row_count,
        truncated=result.truncated or result.row_count > request.max_rows,
        tables_used=result.tables_used,
        repair_attempts=result.repair_attempts,
        guardrail_blocked=result.guardrail_blocked,
        warnings=result.warnings,
        attempts=[
            AttemptInfo(
                attempt=a.attempt,
                outcome=a.outcome,
                sql=a.sql,
                error=a.error,
                violations=a.violations,
            )
            for a in (result.attempts if request.explain else [])
        ],
        usage=Usage(
            provider=result.provider,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=round(result.cost_usd, 6),
        ),
        timings_ms=Timings(
            retrieval=round(result.retrieval_ms, 1),
            generation=round(result.generation_ms, 1),
            validation=round(result.validation_ms, 1),
            execution=round(result.execution_ms, 1),
            answer=round(result.answer_ms, 1),
            total=round(result.total_ms, 1),
        ),
        log_id=result.log_id,
    )


@app.post("/validate", response_model=ValidateResponse, tags=["query"])
def validate(request: ValidateRequest) -> ValidateResponse:
    """Check SQL against the guardrails without running it."""
    result = _require_pipeline().validator.validate(request.sql)
    return ValidateResponse(
        accepted=result.is_valid,
        sql=result.sql,
        original_sql=result.original_sql,
        violations=result.violations,
        warnings=result.warnings,
        tables_referenced=result.tables_referenced,
        limit_applied=result.limit_applied,
        limit_was_injected=result.limit_was_injected,
    )


@app.get("/schema", response_model=SchemaResponse, tags=["schema"])
def schema(
    question: str = Query(
        default="",
        description="If given, return only what the retriever selects for this question.",
    ),
) -> SchemaResponse:
    """The database schema, exactly as the model receives it."""
    pipeline = _require_pipeline()
    catalog = pipeline.catalog

    if question:
        link = pipeline.generator.linker.link(question)
        rendered = catalog.render_schema(link.tables, link.columns)
        if hints := link.render_value_hints():
            rendered += "\n\n" + hints
        names = link.tables
    else:
        rendered, names = catalog.render_schema(), catalog.table_names

    return SchemaResponse(
        tables=[
            TableSummary(
                name=name,
                kind=catalog.tables[name].kind,
                row_count=catalog.tables[name].row_count,
                column_count=len(catalog.tables[name].columns),
                description=catalog.tables[name].description,
            )
            for name in names
            if name in catalog.tables
        ],
        total_rows=sum(
            t.row_count for t in catalog.tables.values() if t.kind == "table"
        ),
        rendered=rendered,
        retrieved_for=question or None,
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness, readiness, and aggregate usage stats."""
    settings = get_settings()
    pipeline = _state.get("pipeline")
    query_log: QueryLog | None = _state.get("query_log")

    ready = pipeline is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=__version__,
        warehouse_ready=ready,
        llm_provider=settings.llm_provider.value,
        llm_model=settings.resolved_model,
        api_key_present=bool(settings.active_api_key),
        tables=len(pipeline.catalog.tables) if pipeline else 0,
        stats=query_log.stats() if query_log else {},
    )


@app.get("/history", response_model=list[HistoryEntry], tags=["ops"])
def history(
    limit: int = Query(default=20, ge=1, le=200),
    failures_only: bool = Query(default=False),
) -> list[HistoryEntry]:
    """Recent questions from the audit log."""
    query_log: QueryLog | None = _state.get("query_log")
    if query_log is None:
        return []

    entries = query_log.failures(limit) if failures_only else query_log.recent(limit)
    return [
        HistoryEntry(
            id=e["id"],
            created_at=e["created_at"],
            question=e["question"],
            sql=e.get("sql"),
            answer=e.get("answer"),
            success=e["success"],
            refused=e.get("refused", False),
            row_count=e.get("row_count"),
            repair_attempts=e.get("repair_attempts", 0),
            total_ms=e.get("total_ms"),
        )
        for e in entries
    ]


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    """A pointer to the docs, so hitting the bare host is not a 404."""
    return {
        "name": "nl2sql-bank",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/ask", "/validate", "/schema", "/health", "/history"],
    }
