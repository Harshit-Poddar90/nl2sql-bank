"""Running validated SQL against the warehouse, safely and with a deadline."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from nl2sql.config import Settings, get_settings
from nl2sql.exceptions import DataError, QueryExecutionError, QueryTimeoutError
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)

#: How often SQLite calls the progress handler, in virtual-machine
#: instructions. Small enough that a runaway query is interrupted promptly,
#: large enough that the callback overhead is irrelevant. At roughly 10M
#: instructions/second this is about a millisecond between checks.
PROGRESS_HANDLER_INSTRUCTIONS = 10_000


@dataclass
class QueryResult:
    """The outcome of running one query."""

    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    sql: str = ""

    #: True when more rows existed than the limit allowed. The caller needs
    #: this: "1,000 accounts" and "at least 1,000 accounts" are different claims.
    truncated: bool = False
    latency_ms: float = 0.0

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        return not self.rows

    @property
    def is_scalar(self) -> bool:
        """A single cell -- the shape of every "how many..." answer."""
        return len(self.rows) == 1 and len(self.columns) == 1

    @property
    def scalar_value(self) -> Any:
        return self.rows[0][0] if self.is_scalar else None

    def to_markdown(self, max_rows: int = 20) -> str:
        """A markdown table, for the CLI and for logs."""
        if not self.columns:
            return "(no results)"
        if self.is_empty:
            return f"| {' | '.join(self.columns)} |\n|{'---|' * len(self.columns)}\n(no rows)"

        lines = [
            f"| {' | '.join(self.columns)} |",
            f"|{'---|' * len(self.columns)}",
        ]
        for row in self.rows[:max_rows]:
            cells = ["NULL" if v is None else str(v) for v in row]
            lines.append(f"| {' | '.join(cells)} |")
        if self.row_count > max_rows:
            lines.append(f"\n_... and {self.row_count - max_rows:,} more rows_")
        return "\n".join(lines)

    def summary(self) -> dict[str, object]:
        return {
            "rows": self.row_count,
            "columns": len(self.columns),
            "truncated": self.truncated,
            "latency_ms": round(self.latency_ms, 1),
        }


class QueryExecutor:
    """Runs validated SQL. One instance per process; the engine pools connections."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._engine = self._build_engine()

    def _build_engine(self) -> Engine:
        """Create a read-only engine for the warehouse."""
        path = self.settings.db_file
        if not path.exists():
            raise DataError(
                f"No warehouse at {path}.",
                user_message="The database has not been built. Run: nl2sql data build",
            )

        def _readonly_connection() -> sqlite3.Connection:
            # `mode=ro` is enforced by SQLite itself: any write raises "attempt
            # to write a readonly database", whatever the SQL says.
            # check_same_thread=False is safe because the pool never hands one
            # connection to two threads at once.
            return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True,
                                   check_same_thread=False)

        # The real URL (not "sqlite://") so SQLAlchemy picks a QueuePool. An
        # in-memory URL gets SingletonThreadPool, which closes other threads'
        # connections once more than five threads are active.
        return create_engine(self.settings.db_url, creator=_readonly_connection)

    def execute(self, sql: str, *, timeout: float | None = None,
                max_rows: int | None = None) -> QueryResult:
        """Run ``sql`` and return the results."""
        timeout = timeout or self.settings.query_timeout_seconds
        max_rows = max_rows or self.settings.max_result_rows
        started = time.perf_counter()

        try:
            with self._engine.connect() as connection:
                deadline = time.monotonic() + timeout
                self._arm_timeout(connection, deadline)

                try:
                    cursor = connection.execute(text(sql))

                    columns = list(cursor.keys())
                    # One more than the cap, so a full page tells us whether
                    # anything was left behind.
                    fetched = cursor.fetchmany(max_rows + 1)
                    truncated = len(fetched) > max_rows
                    rows = [tuple(row) for row in fetched[:max_rows]]

                finally:
                    self._disarm_timeout(connection)

            result = QueryResult(
                columns=columns,
                rows=rows,
                sql=sql,
                truncated=truncated,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            log.info("query_executed", extra=result.summary())
            return result

        except SQLAlchemyError as exc:
            elapsed = time.perf_counter() - started
            message = str(getattr(exc, "orig", exc))

            # SQLite reports an aborted statement as "interrupted". Distinguish
            # that from a genuine SQL error, because the two need completely
            # different repair advice.
            if "interrupted" in message.lower() or elapsed >= timeout * 0.95:
                log.warning("query_timeout", extra={"seconds": round(elapsed, 1)})
                raise QueryTimeoutError(
                    f"Query exceeded the {timeout:g}s limit and was cancelled.",
                    sql=sql,
                    db_message=message,
                ) from exc

            log.warning("query_failed", extra={"error": message[:200]})
            raise QueryExecutionError(
                f"The database rejected the query: {message}",
                sql=sql,
                db_message=message,
            ) from exc

    @staticmethod
    def _arm_timeout(connection: Any, deadline: float) -> None:
        """Install a SQLite progress handler that aborts once past ``deadline``."""
        raw = connection.connection.dbapi_connection
        def _abort_if_expired() -> int:
            return 1 if time.monotonic() > deadline else 0

        raw.set_progress_handler(_abort_if_expired, PROGRESS_HANDLER_INSTRUCTIONS)

    @staticmethod
    def _disarm_timeout(connection: Any) -> None:
        """Remove the progress handler so a pooled connection is not left armed."""
        connection.connection.dbapi_connection.set_progress_handler(None, 0)

    def explain(self, sql: str) -> list[str]:
        """Return the query plan. Used by the CLI's ``--explain`` flag."""
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(text(f"EXPLAIN QUERY PLAN {sql}")).fetchall()
            return [" ".join(str(cell) for cell in row) for row in rows]
        except SQLAlchemyError as exc:
            return [f"(could not explain: {exc})"]

    def dispose(self) -> None:
        self._engine.dispose()
