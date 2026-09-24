"""An append-only audit trail of every question the system was asked."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nl2sql.config import Settings, get_settings
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT    NOT NULL,

    question            TEXT    NOT NULL,
    sql                 TEXT,
    answer              TEXT,

    success             INTEGER NOT NULL,
    error_type          TEXT,
    error_message       TEXT,
    refused             INTEGER NOT NULL DEFAULT 0,

    tables_used         TEXT,
    repair_attempts     INTEGER NOT NULL DEFAULT 0,
    guardrail_blocked   INTEGER NOT NULL DEFAULT 0,
    guardrail_violations TEXT,

    row_count           INTEGER,
    truncated           INTEGER NOT NULL DEFAULT 0,

    provider            TEXT,
    model               TEXT,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cost_usd            REAL    NOT NULL DEFAULT 0,

    retrieval_ms        REAL,
    generation_ms       REAL,
    validation_ms       REAL,
    execution_ms        REAL,
    answer_ms           REAL,
    total_ms            REAL
);

CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log (created_at);
CREATE INDEX IF NOT EXISTS idx_query_log_success ON query_log (success);
"""


class QueryLog:
    """Writes and queries the audit trail."""

    def __init__(self, path: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = path or self.settings.query_log_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialise()

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self):  # type: ignore[no-untyped-def]
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def record(self, result: Any) -> str:
        """Append one ``AskResult`` and return its id."""
        timings = ("retrieval", "generation", "validation", "execution", "answer", "total")
        row = {
            "id": uuid.uuid4().hex[:16],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question": result.question,
            "sql": result.sql or None,
            "answer": result.answer or None,
            "success": int(result.success),
            "error_type": result.error_type,
            "error_message": result.error,
            "refused": int(result.refused),
            "tables_used": json.dumps(result.tables_used),
            "repair_attempts": result.repair_attempts,
            "guardrail_blocked": int(result.guardrail_blocked),
            "guardrail_violations": json.dumps([v for a in result.attempts for v in a.violations]),
            "row_count": result.row_count,
            "truncated": int(result.truncated),
            "provider": result.provider or None,
            "model": result.model or None,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            **{f"{t}_ms": getattr(result, f"{t}_ms") for t in timings},
        }
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    f"INSERT INTO query_log ({', '.join(row)}) "
                    f"VALUES ({', '.join(':' + key for key in row)})",
                    row,
                )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("query_log_write_failed", extra={"error": str(exc)})
        return row["id"]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """The most recent entries, newest first."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM query_log ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._decode(dict(row)) for row in rows]
        except Exception as exc:  # pragma: no cover
            log.warning("query_log_read_failed", extra={"error": str(exc)})
            return []

    def stats(self) -> dict[str, Any]:
        """Aggregate health metrics. Served by ``/health`` and shown in the UI."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT
                        COUNT(*)                                     AS total,
                        SUM(success)                                 AS successful,
                        SUM(guardrail_blocked)                       AS blocked,
                        SUM(refused)                                 AS refused,
                        SUM(CASE WHEN repair_attempts > 0 THEN 1 END) AS repaired,
                        AVG(total_ms)                                AS avg_ms,
                        SUM(input_tokens + output_tokens)            AS tokens,
                        SUM(cost_usd)                                AS cost
                    FROM query_log
                    """
                ).fetchone()

            total = row["total"] or 0
            successful = row["successful"] or 0
            return {
                "total_queries": total,
                "successful": successful,
                "success_rate": round(successful / total, 4) if total else None,
                "guardrail_blocked": row["blocked"] or 0,
                "refused": row["refused"] or 0,
                "needed_repair": row["repaired"] or 0,
                "avg_latency_ms": round(row["avg_ms"], 1) if row["avg_ms"] else None,
                "total_tokens": row["tokens"] or 0,
                "total_cost_usd": round(row["cost"] or 0.0, 4),
            }
        except Exception as exc:  # pragma: no cover
            log.warning("query_log_stats_failed", extra={"error": str(exc)})
            return {"total_queries": 0}

    def failures(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent failures -- the raw material for new benchmark cases."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM query_log WHERE success = 0 "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._decode(dict(row)) for row in rows]
        except Exception:  # pragma: no cover
            return []

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        """Undo :meth:`QueryLogEntry.to_row`."""
        for key in ("tables_used", "guardrail_violations"):
            if isinstance(row.get(key), str):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    row[key] = []
        for key in ("success", "refused", "guardrail_blocked", "truncated"):
            if key in row:
                row[key] = bool(row[key])
        return row
