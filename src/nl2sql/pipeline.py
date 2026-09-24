"""The pipeline: question in, answer out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from nl2sql.answer.synthesizer import AnswerSynthesizer
from nl2sql.catalog.catalog import Catalog, build_catalog
from nl2sql.config import Settings, get_settings
from nl2sql.exceptions import (
    GuardrailViolation,
    NL2SQLError,
    QueryExecutionError,
    SQLGenerationError,
)
from nl2sql.execution.executor import QueryExecutor, QueryResult
from nl2sql.generation.generator import GenerationResult, SQLGenerator
from nl2sql.guardrails.validator import SQLValidator, ValidationResult
from nl2sql.llm.base import LLMClient
from nl2sql.llm.factory import create_llm_client
from nl2sql.logging_setup import get_logger
from nl2sql.observability.query_log import QueryLog
from nl2sql.retrieval.schema_linker import SchemaLinker

log = get_logger(__name__)


@dataclass
class AttemptRecord:
    """What happened on one pass through generate -> validate -> execute."""

    attempt: int
    sql: str
    outcome: str                      # "success" | "guardrail" | "execution" | "generation"
    error: str | None = None
    violations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class AskResult:
    """Everything the pipeline produced for one question."""

    question: str
    answer: str = ""
    sql: str = ""

    success: bool = False
    refused: bool = False
    refusal_reason: str | None = None
    error: str | None = None
    error_type: str | None = None

    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False

    tables_used: list[str] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    validation_ms: float = 0.0
    execution_ms: float = 0.0
    answer_ms: float = 0.0
    total_ms: float = 0.0

    log_id: str = ""

    @property
    def repair_attempts(self) -> int:
        """How many times we had to go back to the model. 0 means first-try."""
        return max(0, len(self.attempts) - 1)

    @property
    def guardrail_blocked(self) -> bool:
        return any(a.outcome == "guardrail" for a in self.attempts)

    def to_dict(self, *, include_rows: bool = True) -> dict[str, Any]:
        """JSON-serialisable form, for the API."""
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "sql": self.sql,
            "success": self.success,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "error": self.error,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "tables_used": self.tables_used,
            "repair_attempts": self.repair_attempts,
            "guardrail_blocked": self.guardrail_blocked,
            "warnings": self.warnings,
            "usage": {
                "provider": self.provider,
                "model": self.model,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6),
            },
            "timings_ms": {
                "retrieval": round(self.retrieval_ms, 1),
                "generation": round(self.generation_ms, 1),
                "validation": round(self.validation_ms, 1),
                "execution": round(self.execution_ms, 1),
                "answer": round(self.answer_ms, 1),
                "total": round(self.total_ms, 1),
            },
            "log_id": self.log_id,
        }
        if include_rows:
            payload["columns"] = self.columns
            payload["rows"] = [list(row) for row in self.rows]
        return payload


class Pipeline:
    """Wires the stages together and runs them."""

    def __init__(
        self,
        catalog: Catalog,
        generator: SQLGenerator,
        validator: SQLValidator,
        executor: QueryExecutor,
        synthesizer: AnswerSynthesizer,
        query_log: QueryLog | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog
        self.generator = generator
        self.validator = validator
        self.executor = executor
        self.synthesizer = synthesizer
        self.query_log = query_log

    @classmethod
    def build(
        cls,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        enable_logging: bool = True,
        fallback_to_stub: bool = False,
    ) -> Pipeline:
        """Construct a pipeline with everything wired up."""
        settings = settings or get_settings()
        started = time.perf_counter()

        catalog = build_catalog(settings)
        linker = SchemaLinker(catalog, settings)
        client = llm or create_llm_client(settings, fallback_to_stub=fallback_to_stub)

        pipeline = cls(
            catalog=catalog,
            generator=SQLGenerator(catalog, linker, client, settings=settings),
            validator=SQLValidator(catalog, settings),
            executor=QueryExecutor(settings),
            synthesizer=AnswerSynthesizer(client, settings),
            query_log=QueryLog(settings=settings) if enable_logging else None,
            settings=settings,
        )

        log.info(
            "pipeline_ready",
            extra={
                "provider": client.provider,
                "model": client.model,
                "tables": len(catalog.tables),
                "startup_seconds": round(time.perf_counter() - started, 2),
            },
        )
        return pipeline

    # -----------------------------------------------------------------------
    def ask(self, question: str, *, force_llm_answer: bool = False) -> AskResult:
        """Answer a question. The one method callers need."""
        started = time.perf_counter()
        result = AskResult(question=question.strip())

        if not result.question:
            result.error = "The question is empty."
            result.error_type = "ValidationError"
            return result

        try:
            self._run(result, force_llm_answer=force_llm_answer)
        except NL2SQLError as exc:
            result.error = exc.user_message
            result.error_type = type(exc).__name__
            log.warning("pipeline_failed",
                        extra={"error_type": result.error_type, "error": exc.message[:200]})
        except Exception as exc:  # pragma: no cover - genuinely unexpected
            result.error = "An unexpected error occurred."
            result.error_type = type(exc).__name__
            log.exception("pipeline_crashed", extra={"error": str(exc)[:200]})

        result.total_ms = (time.perf_counter() - started) * 1000
        # Priced once from the final totals, so repairs and answer synthesis are included.
        price_in, price_out = self.settings.price_per_mtok()
        result.cost_usd = (result.input_tokens * price_in + result.output_tokens * price_out) / 1e6
        if self.query_log is not None:
            result.log_id = self.query_log.record(result)
        return result

    def _run(self, result: AskResult, *, force_llm_answer: bool = False) -> None:
        """Generate, validate, execute, repair, answer."""
        # -- [1] + [2] first attempt
        generation_started = time.perf_counter()
        generation = self.generator.generate(result.question)
        result.generation_ms += (time.perf_counter() - generation_started) * 1000
        result.retrieval_ms = generation.link.latency_ms
        result.tables_used = list(generation.link.tables)
        self._account(result, generation)

        if generation.refused:
            result.refused = True
            result.refusal_reason = generation.refusal_reason
            result.answer = generation.refusal_reason or "That cannot be answered from this data."
            result.sql = ""
            log.info("question_refused", extra={"reason": result.refusal_reason})
            return

        # -- [3]..[6] validate, execute, repair
        query_result = self._validate_execute_repair(result, generation)
        if query_result is None:
            return

        # -- [5] answer synthesis
        answer_started = time.perf_counter()
        answer = self.synthesizer.synthesize(
            result.question, result.sql, query_result, force_llm=force_llm_answer
        )
        result.answer_ms = (time.perf_counter() - answer_started) * 1000
        result.answer = answer.text

        if answer.llm_response is not None:
            result.input_tokens += answer.llm_response.input_tokens
            result.output_tokens += answer.llm_response.output_tokens

        result.success = True

    def _validate_execute_repair(
        self, result: AskResult, generation: GenerationResult
    ) -> QueryResult | None:
        """The repair loop. Returns the query result, or None if we gave up."""
        max_attempts = self.settings.max_repair_attempts + 1
        current = generation

        for attempt in range(max_attempts):
            attempt_started = time.perf_counter()
            sql = current.sql
            result.sql = sql

            try:
                # -- [3] guardrails
                validation_started = time.perf_counter()
                validation: ValidationResult = self.validator.validate(sql)
                result.validation_ms += (time.perf_counter() - validation_started) * 1000
                validation.raise_if_invalid()

                result.warnings.extend(validation.warnings)
                # Run the *rewritten* query -- the one with the row limit
                # applied -- never the model's original text.
                result.sql = validation.sql

                # -- [4] execution
                execution_started = time.perf_counter()
                query_result = self.executor.execute(validation.sql)
                result.execution_ms += (time.perf_counter() - execution_started) * 1000

                result.columns = query_result.columns
                result.rows = query_result.rows
                result.row_count = query_result.row_count
                result.truncated = query_result.truncated

                result.attempts.append(
                    AttemptRecord(
                        attempt=attempt,
                        sql=validation.sql,
                        outcome="success",
                        latency_ms=(time.perf_counter() - attempt_started) * 1000,
                    )
                )
                return query_result

            except (GuardrailViolation, QueryExecutionError) as exc:
                outcome = "guardrail" if isinstance(exc, GuardrailViolation) else "execution"
                result.attempts.append(
                    AttemptRecord(
                        attempt=attempt,
                        sql=sql,
                        outcome=outcome,
                        error=exc.message,
                        violations=getattr(exc, "violations", []),
                        latency_ms=(time.perf_counter() - attempt_started) * 1000,
                    )
                )

                is_last = attempt >= max_attempts - 1
                if is_last:
                    result.error = exc.user_message
                    result.error_type = type(exc).__name__
                    log.warning(
                        "repair_exhausted",
                        extra={"attempts": attempt + 1, "outcome": outcome},
                    )
                    return None

                # -- [6] repair: hand the model its own error and try again.
                log.info("repairing", extra={"attempt": attempt + 1, "reason": outcome})
                try:
                    generation_started = time.perf_counter()
                    current = self.generator.repair(
                        result.question,
                        sql,
                        exc.feedback_for_model(),
                        current,
                        attempt=attempt + 1,
                    )
                    result.generation_ms += (time.perf_counter() - generation_started) * 1000
                    self._account(result, current)

                    if current.refused:
                        result.refused = True
                        result.refusal_reason = current.refusal_reason
                        result.answer = current.refusal_reason or ""
                        return None

                except SQLGenerationError as repair_exc:
                    result.error = repair_exc.user_message
                    result.error_type = type(repair_exc).__name__
                    return None

        return None

    @staticmethod
    def _account(result: AskResult, generation: GenerationResult) -> None:
        """Accumulate token usage and provider metadata across every attempt."""
        response = generation.llm_response
        if response is None:
            return
        result.provider = response.provider
        result.model = response.model
        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens

    def close(self) -> None:
        self.executor.dispose()
