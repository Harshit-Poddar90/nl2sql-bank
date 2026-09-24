"""The safety validator: everything that stands between a model and the database."""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from nl2sql.catalog.catalog import Catalog
from nl2sql.config import Settings, get_settings
from nl2sql.exceptions import GuardrailViolation
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)

#: Statement types that write, change structure, or reach outside the query
#: engine. Checked by AST node class, so spelling and formatting are irrelevant.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Merge,
    # sqlglot parses statements it has no dedicated node for -- PRAGMA, ATTACH,
    # VACUUM, and anything else vendor-specific -- as a generic Command. There
    # is no legitimate reason for one to appear in a generated SELECT, so the
    # whole class is refused.
    exp.Command,
)

#: Functions that touch the filesystem, load native code, or otherwise escape
#: the query sandbox. None of these can appear in an analytical query.
FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        "load_extension",   # SQLite: loads arbitrary native code. Game over.
        "readfile",         # SQLite CLI: reads any file the process can read
        "writefile",        # SQLite CLI: writes anywhere the process can write
        "edit",
        "pg_read_file",     # Postgres equivalents, for when the URL points there
        "pg_ls_dir",
        "lo_import",
        "lo_export",
        "dblink",
        "copy",
        "system",
        "shell",
        "eval",
    }
)

DIALECT = "sqlite"


@dataclass
class ValidationResult:
    """The verdict, and the query that is safe to run."""

    #: The query after rewriting -- LIMIT injected, formatting normalised.
    #: This is what the executor must run; never run the original.
    sql: str
    original_sql: str

    #: Problems that caused rejection. Non-empty means ``is_valid`` is False.
    violations: list[str] = field(default_factory=list)
    #: Concerns that did not justify rejection. Surfaced for observability.
    warnings: list[str] = field(default_factory=list)

    tables_referenced: list[str] = field(default_factory=list)
    columns_referenced: list[str] = field(default_factory=list)
    limit_applied: int | None = None
    limit_was_injected: bool = False
    latency_ms: float = 0.0

    @property
    def is_valid(self) -> bool:
        return not self.violations

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise GuardrailViolation(
                f"Query rejected: {'; '.join(self.violations)}",
                violations=self.violations,
                sql=self.original_sql,
            )


class SQLValidator:
    """Validates and rewrites generated SQL. Stateless and cheap; reuse freely."""

    def __init__(self, catalog: Catalog, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog
        self._allowed_tables, self._allowed_columns = catalog.identifier_allowlist()

    # -- entry point --------------------------------------------------------
    def validate(self, sql: str) -> ValidationResult:
        """Check ``sql`` and return the safe, rewritten version."""
        started = time.perf_counter()
        result = ValidationResult(sql=sql, original_sql=sql)
        self._validate(sql, result)
        result.latency_ms = (time.perf_counter() - started) * 1000
        log.debug("sql_validated", extra={"valid": result.is_valid, "tables": result.tables_referenced})
        return result

    def _validate(self, sql: str, result: ValidationResult) -> None:
        # -- 1. Does it parse, and is it exactly one statement?
        try:
            statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
        except ParseError as exc:
            result.violations.append(f"The query is not valid SQLite: {exc}")
            return

        if len(statements) != 1:
            # More than one is the classic stacked-injection shape. Refuse
            # outright rather than running "just the first one".
            result.violations.append(
                "The query is empty." if not statements else
                f"Expected a single statement but found {len(statements)}. "
                f"Multiple statements are never allowed."
            )
            return

        statement = statements[0]

        # -- 2..5. Content checks. All of them run even after one fails, so the
        # repair prompt gets the complete list rather than one problem at a time.
        self._check_read_only(statement, result)
        self._check_forbidden_functions(statement, result)
        self._check_identifiers(statement, result)
        self._check_join_sanity(statement, result)

        # -- 6. Rewrite. For a UNION the root node is replaced, so render what comes back.
        if result.is_valid:
            statement = self._apply_row_limit(statement, result)
            result.sql = statement.sql(dialect=DIALECT, pretty=True)

    # -- individual checks --------------------------------------------------
    def _check_read_only(self, statement: exp.Expr, result: ValidationResult) -> None:
        """Reject anything that is not a pure read."""
        # The statement itself must be a query.
        allowed_roots = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)
        if not isinstance(statement, allowed_roots):
            result.violations.append(
                f"Only SELECT queries are allowed; this is a "
                f"{type(statement).__name__.upper()} statement."
            )

        # ...and nothing anywhere inside it may write. A subquery, a CTE, or a
        # set-operation branch is just as dangerous as a top-level write.
        for node_type in FORBIDDEN_NODES:
            found = list(statement.find_all(node_type))
            if found:
                result.violations.append(
                    f"{node_type.__name__.upper()} is not permitted. "
                    f"This system has read-only access to the database."
                )

    def _check_forbidden_functions(
        self, statement: exp.Expr, result: ValidationResult
    ) -> None:
        """Reject filesystem and extension-loading functions."""
        for node in statement.find_all(exp.Func):
            # exp.Anonymous covers functions sqlglot has no dedicated node for,
            # which is where all the dangerous ones live.
            name = node.name if isinstance(node, exp.Anonymous) else type(node).__name__
            if name.lower() in FORBIDDEN_FUNCTIONS:
                result.violations.append(f"The function {name.lower()}() is not permitted.")

    def _check_identifiers(
        self, statement: exp.Expr, result: ValidationResult
    ) -> None:
        """Every table and column must exist. This is the anti-hallucination check."""
        ctes = {cte.alias.lower() for cte in statement.find_all(exp.CTE) if cte.alias}
        aliases = ctes | {
            node.alias.lower()
            for node in statement.find_all(exp.Table, exp.Subquery, exp.Alias)
            if node.alias
        }

        unknown_tables: list[str] = []
        for table in statement.find_all(exp.Table):
            name = table.name.lower()
            if name in self._allowed_tables:
                canonical = next(t for t in self.catalog.tables if t.lower() == name)
                if canonical not in result.tables_referenced:
                    result.tables_referenced.append(canonical)
            elif name and name not in ctes:
                unknown_tables.append(table.name)

        unknown_columns: list[str] = []
        for column in statement.find_all(exp.Column):
            name = column.name.lower()
            if name in self._allowed_columns or name in aliases:
                if column.name not in result.columns_referenced:
                    result.columns_referenced.append(column.name)
            elif name and name != "*":
                unknown_columns.append(column.name)

        for kind, unknown, allowed in (("table", unknown_tables, self._allowed_tables),
                                       ("column", unknown_columns, self._allowed_columns)):
            if unknown:
                rendered = ", ".join(
                    f"'{name}'" + (f" (did you mean '{hint}'?)" if hint else "")
                    for name in dict.fromkeys(unknown)
                    for hint in [self._closest(name.lower(), allowed)]
                )
                result.violations.append(f"Unknown {kind}(s): {rendered}")

    #: A table with at least this many rows makes any cross join involving it
    #: unacceptable rather than merely suspicious.
    LARGE_TABLE_ROWS = 100_000

    def _check_join_sanity(
        self, statement: exp.Expr, result: ValidationResult
    ) -> None:
        """Catch runaway joins before the database spends an hour on one."""
        table_names = {t.name.lower() for t in statement.find_all(exp.Table) if t.name}
        table_count = len(table_names)

        if table_count > self.settings.max_join_tables:
            result.violations.append(
                f"The query joins {table_count} tables, above the limit of "
                f"{self.settings.max_join_tables}. Simplify it."
            )

        large_tables = {
            name
            for name in table_names
            if (table := self.catalog.table(name)) is not None
            and table.row_count >= self.LARGE_TABLE_ROWS
        }

        for join in statement.find_all(exp.Join):
            if join.args.get("on") is not None or join.args.get("using"):
                continue

            message = (
                "A join has no ON condition, so it produces a cross product of "
                "every row in both tables. Add the join condition."
            )

            if large_tables:
                result.violations.append(
                    f"{message} This query cross-joins "
                    f"{', '.join(sorted(large_tables))}, which would generate an "
                    f"enormous result set."
                )
            elif table_count >= 3:
                result.violations.append(message)
            else:
                result.warnings.append(message)
            return  # one report is enough; they all have the same cause

    def _apply_row_limit(
        self, statement: exp.Expr, result: ValidationResult
    ) -> exp.Expr:
        """Guarantee a bounded result set by editing the AST."""
        maximum = self.settings.max_result_rows

        if isinstance(statement, exp.Select):
            existing = statement.args.get("limit")

            if existing is None:
                statement.set("limit", exp.Limit(expression=exp.Literal.number(maximum)))
                result.limit_applied = maximum
                result.limit_was_injected = True
                return statement

            try:
                current = int(existing.expression.name)
            except (AttributeError, ValueError):
                # A non-literal LIMIT (a parameter, or an expression). We cannot
                # reason about its value, so replace it with one we can.
                statement.set("limit", exp.Limit(expression=exp.Literal.number(maximum)))
                result.limit_applied = maximum
                result.limit_was_injected = True
                result.warnings.append(
                    "The query's LIMIT was not a plain number and has been replaced."
                )
                return statement

            if current > maximum:
                existing.set("expression", exp.Literal.number(maximum))
                result.limit_applied = maximum
                result.limit_was_injected = True
                result.warnings.append(
                    f"The query asked for {current:,} rows; capped at {maximum:,}."
                )
            else:
                result.limit_applied = current
            return statement

        # UNION / INTERSECT / EXCEPT cannot reliably take a LIMIT directly
        # across dialects, so wrap the whole thing:
        #     SELECT * FROM (<query>) AS _bounded LIMIT n
        if isinstance(statement, (exp.Union, exp.Except, exp.Intersect)):
            wrapped = exp.Select().select("*").from_(
                exp.Subquery(
                    this=statement.copy(),
                    alias=exp.TableAlias(this=exp.Identifier(this="_bounded")),
                )
            )
            wrapped.set("limit", exp.Limit(expression=exp.Literal.number(maximum)))
            result.limit_applied = maximum
            result.limit_was_injected = True
            return wrapped

        return statement

    @staticmethod
    def _closest(name: str, candidates: set[str]) -> str | None:
        """Best near-match for an unknown identifier, or None."""
        matches = difflib.get_close_matches(name, sorted(candidates), n=1, cutoff=0.75)
        return matches[0] if matches else None
