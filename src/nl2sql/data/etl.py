"""Transform the raw Berka CSVs into the query-ready warehouse."""

from __future__ import annotations

import csv
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, MetaData, Table, create_engine, text

from nl2sql.config import Settings, get_settings
from nl2sql.data.download import download_dataset
from nl2sql.exceptions import DataError
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)

# The source CSVs use ';' and quote their headers. A million-row file also
# needs the field-size limit raised, because csv defaults to 128 KB per field
# and errors out rather than warning if a quote goes unbalanced.
CSV_DELIMITER = ";"
BATCH_SIZE = 50_000

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:  # pragma: no cover - 32-bit platforms
    csv.field_size_limit(2**31 - 1)


# ===========================================================================
# Value translation
#
# Every mapping the raw data needs, in one place, so that when you find a
# Czech string in a result you have exactly one file to check.
# ===========================================================================

#: account.frequency -- how often the bank issues a statement.
STATEMENT_FREQUENCY: dict[str, str] = {
    "POPLATEK MESICNE": "monthly",
    "POPLATEK TYDNE": "weekly",
    "POPLATEK PO OBRATU": "after each transaction",
}

#: trans.type -- which way the money moved.
TRANSACTION_DIRECTION: dict[str, str] = {
    "PRIJEM": "credit",
    "VYDAJ": "withdrawal",
    # 'VYBER' appears as a type in some rows as well as an operation. It always
    # means money leaving the account.
    "VYBER": "withdrawal",
}

#: trans.operation -- the mechanism used.
TRANSACTION_OPERATION: dict[str, str] = {
    "VKLAD": "cash deposit",
    "VYBER": "cash withdrawal",
    "PREVOD Z UCTU": "collection from another bank",
    "PREVOD NA UCET": "remittance to another bank",
    "VYBER KARTOU": "credit card withdrawal",
}

#: trans.k_symbol / order.k_symbol -- what the payment was for.
PAYMENT_PURPOSE: dict[str, str] = {
    "POJISTNE": "insurance payment",
    "SLUZBY": "statement fee",
    "UROK": "interest credited",
    "SANKC. UROK": "penalty interest for negative balance",
    "SIPO": "household payment",
    "DUCHOD": "old-age pension",
    "UVER": "loan payment",
    "LEASING": "leasing payment",
}

#: loan.status -- the outcome of the loan. This is the dataset's target variable.
LOAN_STATUS: dict[str, str] = {
    "A": "finished, paid in full",
    "B": "finished, not paid (defaulted)",
    "C": "running, payments up to date",
    "D": "running, client in debt",
}

#: Loans in state A or B have run their course.
FINISHED_STATUSES = frozenset({"A", "B"})
#: B defaulted outright; D is currently in arrears. Both count as "in trouble".
DEFAULTED_STATUSES = frozenset({"B", "D"})


def clean(value: str | None) -> str | None:
    """Normalise a raw CSV cell."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped in {"", "?", "-"}:
        return None
    return stripped


def parse_date(raw: str | None) -> str | None:
    """Convert a Berka ``YYMMDD`` date into ISO ``'YYYY-MM-DD'``."""
    value = clean(raw)
    if value is None:
        return None

    digits = value.split()[0]          # drop any ' 00:00:00' tail
    digits = digits.zfill(6)           # a leading zero can be lost upstream

    if len(digits) != 6 or not digits.isdigit():
        raise ValueError(f"Cannot parse Berka date {raw!r}")

    year = 1900 + int(digits[0:2])
    month = int(digits[2:4])
    day = int(digits[4:6])

    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError(f"Date {raw!r} decodes to an impossible {year}-{month}-{day}")

    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_birth_number(raw: str) -> tuple[str, str]:
    """Decode a Czech ``birth_number`` into ``(birth_date, gender)``."""
    digits = str(raw).strip().zfill(6)
    if len(digits) != 6 or not digits.isdigit():
        raise ValueError(f"Malformed birth_number {raw!r}")

    year = 1900 + int(digits[0:2])
    month = int(digits[2:4])
    day = int(digits[4:6])

    if month > 50:
        gender = "female"
        month -= 50
    else:
        gender = "male"

    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError(f"birth_number {raw!r} decodes to an impossible date")

    return f"{year:04d}-{month:02d}-{day:02d}", gender


def age_at(birth_date: str, reference: str = "1999-01-01") -> int:
    """Whole years between two ISO dates."""
    by, bm, bd = (int(p) for p in birth_date.split("-"))
    ry, rm, rd = (int(p) for p in reference.split("-"))
    years = ry - by
    if (rm, rd) < (bm, bd):
        years -= 1
    return years


def to_float(raw: str | None) -> float | None:
    value = clean(raw)
    return None if value is None else float(value)


def to_int(raw: str | None) -> int | None:
    value = clean(raw)
    if value is None:
        return None
    # A few numeric columns arrive as '1204953.0'; int() refuses those directly.
    return int(float(value))


# ===========================================================================
# Per-table transforms
#
# Each function takes one raw CSV row (a dict of strings) and returns the row
# to insert, or raises if the row is unusable. Keeping them tiny and separate
# means a failure message can name the exact table and line.
# ===========================================================================

def _row_district(r: dict[str, str]) -> dict[str, Any]:
    """A1..A16 -> named columns."""
    return {
        "district_id": to_int(r["A1"]),
        "district_name": clean(r["A2"]),
        "region": clean(r["A3"]),
        "inhabitants": to_int(r["A4"]),
        "municipalities_under_500": to_int(r["A5"]),
        "municipalities_500_to_1999": to_int(r["A6"]),
        "municipalities_2000_to_9999": to_int(r["A7"]),
        "municipalities_over_10000": to_int(r["A8"]),
        "n_cities": to_int(r["A9"]),
        "urban_ratio_pct": to_float(r["A10"]),
        "average_salary": to_int(r["A11"]),
        "unemployment_rate_1995": to_float(r["A12"]),
        "unemployment_rate_1996": to_float(r["A13"]),
        "entrepreneurs_per_1000": to_int(r["A14"]),
        "crimes_1995": to_int(r["A15"]),
        "crimes_1996": to_int(r["A16"]),
    }


def _row_client(r: dict[str, str]) -> dict[str, Any]:
    birth_date, gender = parse_birth_number(r["birth_number"])
    return {
        "client_id": to_int(r["client_id"]),
        "district_id": to_int(r["district_id"]),
        "birth_number": clean(r["birth_number"]),
        "birth_date": birth_date,
        "gender": gender,
        "age_at_1999": age_at(birth_date),
    }


def _row_account(r: dict[str, str]) -> dict[str, Any]:
    code = clean(r["frequency"]) or ""
    return {
        "account_id": to_int(r["account_id"]),
        "district_id": to_int(r["district_id"]),
        "statement_frequency_code": code,
        # An unmapped code falls through as itself rather than becoming NULL:
        # better a Czech string in the output than a silently dropped fact.
        "statement_frequency": STATEMENT_FREQUENCY.get(code, code.lower()),
        "opened_date": parse_date(r["date"]),
    }


def _row_disposition(r: dict[str, str]) -> dict[str, Any]:
    return {
        "disposition_id": to_int(r["disp_id"]),
        "client_id": to_int(r["client_id"]),
        "account_id": to_int(r["account_id"]),
        "disposition_type": (clean(r["type"]) or "").upper(),
    }


def _row_card(r: dict[str, str]) -> dict[str, Any]:
    return {
        "card_id": to_int(r["card_id"]),
        "disposition_id": to_int(r["disp_id"]),
        "card_type": (clean(r["type"]) or "").lower(),
        "issued_date": parse_date(r["issued"]),
    }


def _row_loan(r: dict[str, str]) -> dict[str, Any]:
    status_code = (clean(r["status"]) or "").upper()
    return {
        "loan_id": to_int(r["loan_id"]),
        "account_id": to_int(r["account_id"]),
        "loan_date": parse_date(r["date"]),
        "amount": to_float(r["amount"]),
        "duration_months": to_int(r["duration"]),
        "monthly_payment": to_float(r["payments"]),
        "status_code": status_code,
        "status": LOAN_STATUS.get(status_code, status_code),
        "is_finished": int(status_code in FINISHED_STATUSES),
        "is_defaulted": int(status_code in DEFAULTED_STATUSES),
    }


def _row_order(r: dict[str, str]) -> dict[str, Any]:
    purpose_code = clean(r["k_symbol"])
    return {
        "order_id": to_int(r["order_id"]),
        "account_id": to_int(r["account_id"]),
        "bank_to": clean(r["bank_to"]),
        "account_to": clean(r["account_to"]),
        "amount": to_float(r["amount"]),
        "purpose_code": purpose_code,
        "purpose": PAYMENT_PURPOSE.get(purpose_code or "", purpose_code),
    }


def _row_transaction(r: dict[str, str]) -> dict[str, Any]:
    direction_code = clean(r["type"]) or ""
    operation_code = clean(r["operation"])
    purpose_code = clean(r["k_symbol"])
    return {
        "transaction_id": to_int(r["trans_id"]),
        "account_id": to_int(r["account_id"]),
        "transaction_date": parse_date(r["date"]),
        "direction_code": direction_code,
        "direction": TRANSACTION_DIRECTION.get(direction_code, direction_code.lower()),
        "operation_code": operation_code,
        "operation": TRANSACTION_OPERATION.get(operation_code or "", operation_code),
        "amount": to_float(r["amount"]),
        "balance_after": to_float(r["balance"]),
        "purpose_code": purpose_code,
        "purpose": PAYMENT_PURPOSE.get(purpose_code or "", purpose_code),
        "partner_bank": clean(r["bank"]),
        "partner_account": clean(r["account"]),
    }


@dataclass(frozen=True)
class TableSpec:
    """How to get one CSV into one warehouse table."""

    csv_name: str
    table_name: str
    transform: Any
    expected_rows: int


# Order matters: parents before children, so foreign keys are satisfiable at
# insert time rather than only at the end.
TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec("district", "district", _row_district, 77),
    TableSpec("client", "client", _row_client, 5_369),
    TableSpec("account", "account", _row_account, 4_500),
    TableSpec("disp", "disposition", _row_disposition, 5_369),
    TableSpec("card", "card", _row_card, 892),
    TableSpec("loan", "loan", _row_loan, 682),
    TableSpec("order", "permanent_order", _row_order, 6_471),
    TableSpec("trans", "bank_transaction", _row_transaction, 1_056_320),
)


def _read_rows(path: Path, spec: TableSpec) -> Iterator[dict[str, Any]]:
    """Stream a CSV through its transform, one row at a time."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=CSV_DELIMITER)
        for line_number, raw in enumerate(reader, start=2):  # line 1 is the header
            try:
                yield spec.transform(raw)
            except Exception as exc:
                raise DataError(
                    f"{path.name} line {line_number}: {exc}",
                    user_message=f"The source file {path.name} contains a row we cannot parse.",
                    details={"line": line_number, "row": raw},
                ) from exc


def _batched(rows: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Group a stream into lists of at most ``size``."""
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# The account_summary rollup
#
# Kept here as a named constant rather than buried in a function so that it is
# greppable: if a number in `account_summary` ever looks wrong, this is the
# definition that produced it.
#
# `latest` uses a window function to find each account's most recent
# transaction. Ties on date are broken by transaction_id, since transactions
# were issued in ascending id order within a day.
# ---------------------------------------------------------------------------
ACCOUNT_SUMMARY_SQL = """
INSERT INTO account_summary (
    account_id, owner_client_id, district_id,
    current_balance, first_transaction_date, last_transaction_date, transaction_count,
    total_credited, total_withdrawn, average_balance, min_balance, max_balance,
    has_card, has_loan, has_defaulted_loan, n_dispositions
)
WITH latest AS (
    SELECT
        account_id,
        balance_after,
        ROW_NUMBER() OVER (
            PARTITION BY account_id
            ORDER BY transaction_date DESC, transaction_id DESC
        ) AS rn
    FROM bank_transaction
),
activity AS (
    SELECT
        account_id,
        COUNT(*)                                                         AS transaction_count,
        MIN(transaction_date)                                            AS first_transaction_date,
        MAX(transaction_date)                                            AS last_transaction_date,
        SUM(CASE WHEN direction = 'credit'     THEN amount ELSE 0 END)   AS total_credited,
        SUM(CASE WHEN direction = 'withdrawal' THEN amount ELSE 0 END)   AS total_withdrawn,
        AVG(balance_after)                                               AS average_balance,
        MIN(balance_after)                                               AS min_balance,
        MAX(balance_after)                                               AS max_balance
    FROM bank_transaction
    GROUP BY account_id
),
owners AS (
    SELECT account_id, MIN(client_id) AS client_id
    FROM disposition
    WHERE disposition_type = 'OWNER'
    GROUP BY account_id
),
disposition_counts AS (
    SELECT account_id, COUNT(*) AS n FROM disposition GROUP BY account_id
),
carded AS (
    SELECT DISTINCT d.account_id
    FROM card c
    JOIN disposition d ON d.disposition_id = c.disposition_id
),
loans AS (
    SELECT account_id, MAX(is_defaulted) AS any_defaulted
    FROM loan
    GROUP BY account_id
)
SELECT
    a.account_id,
    o.client_id,
    a.district_id,
    COALESCE(l.balance_after, 0.0),
    act.first_transaction_date,
    act.last_transaction_date,
    COALESCE(act.transaction_count, 0),
    COALESCE(act.total_credited, 0.0),
    COALESCE(act.total_withdrawn, 0.0),
    act.average_balance,
    act.min_balance,
    act.max_balance,
    CASE WHEN carded.account_id IS NOT NULL THEN 1 ELSE 0 END,
    CASE WHEN loans.account_id  IS NOT NULL THEN 1 ELSE 0 END,
    COALESCE(loans.any_defaulted, 0),
    COALESCE(dc.n, 0)
FROM account a
LEFT JOIN (SELECT account_id, balance_after FROM latest WHERE rn = 1) l
       ON l.account_id = a.account_id
LEFT JOIN activity            act    ON act.account_id    = a.account_id
LEFT JOIN owners              o      ON o.account_id      = a.account_id
LEFT JOIN disposition_counts  dc     ON dc.account_id     = a.account_id
LEFT JOIN carded                     ON carded.account_id = a.account_id
LEFT JOIN loans                      ON loans.account_id  = a.account_id
"""


@dataclass
class LoadReport:
    """What a warehouse build actually did. Printed by the CLI."""

    tables: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    database_path: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.tables.values())

    def render(self) -> str:
        lines = [
            "",
            f"  Warehouse built at {self.database_path}",
            f"  {self.total_rows:,} rows across {len(self.tables)} tables "
            f"in {self.duration_seconds:.1f}s",
            "",
        ]
        width = max((len(name) for name in self.tables), default=10)
        for name, count in self.tables.items():
            lines.append(f"    {name:<{width}}  {count:>10,}")
        if self.warnings:
            lines.append("")
            lines.extend(f"    ! {w}" for w in self.warnings)
        lines.append("")
        return "\n".join(lines)


def _run_outside_transaction(engine: Engine, statements: list[str]) -> None:
    """Execute statements with no transaction open."""
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        for statement in statements:
            connection.exec_driver_sql(statement)


def _bulk_load_pragmas(engine: Engine, *, enable: bool) -> list[str]:
    """The pragmas that trade durability for speed during a load."""
    if engine.dialect.name != "sqlite":
        return []
    if enable:
        return [
            "PRAGMA journal_mode = MEMORY",
            "PRAGMA synchronous = OFF",
            "PRAGMA cache_size = -64000",   # 64 MB page cache, negative = KiB
        ]
    return [
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous = NORMAL",
    ]


def load_warehouse(
    settings: Settings | None = None,
    *,
    force_download: bool = False,
    skip_verification: bool = False,
) -> LoadReport:
    """Build the warehouse from scratch. The one function the CLI calls."""
    settings = settings or get_settings()
    settings.ensure_directories()
    started = time.perf_counter()

    csv_paths = download_dataset(settings, force=force_download)

    if settings.db_file.exists():
        # Rebuild from empty. Simpler and more trustworthy than trying to
        # reconcile an existing file against a changed schema.
        settings.db_file.unlink()
        log.info("existing_database_removed", extra={"path": str(settings.db_file)})

    engine = create_engine(settings.db_url, future=True)
    report = LoadReport(database_path=str(settings.db_file))

    schema_sql = settings.schema_sql_path.read_text(encoding="utf-8")

    _run_outside_transaction(engine, _bulk_load_pragmas(engine, enable=True))

    with engine.begin() as connection:
        # sqlalchemy's `text()` runs one statement at a time; the schema file
        # is a script, so split it. Semicolons only ever terminate statements
        # in this file (no string literals contain one), which we can rely on
        # because we wrote it.
        for statement in _split_sql_script(schema_sql):
            connection.exec_driver_sql(statement)
        log.info("schema_created", extra={"path": str(settings.schema_sql_path)})

        metadata = MetaData()
        metadata.reflect(bind=connection)

        for spec in TABLE_SPECS:
            table: Table = metadata.tables[spec.table_name]
            path = csv_paths[spec.csv_name]
            table_started = time.perf_counter()
            inserted = 0

            for batch in _batched(_read_rows(path, spec), BATCH_SIZE):
                connection.execute(table.insert(), batch)
                inserted += len(batch)
                if inserted % (BATCH_SIZE * 4) == 0:
                    log.debug("loading", extra={"table": spec.table_name, "rows": inserted})

            report.tables[spec.table_name] = inserted
            log.info(
                "table_loaded",
                extra={
                    "table": spec.table_name,
                    "rows": inserted,
                    "seconds": round(time.perf_counter() - table_started, 2),
                },
            )

            if inserted != spec.expected_rows:
                message = (
                    f"{spec.table_name}: loaded {inserted:,} rows but the Berka dataset "
                    f"publishes {spec.expected_rows:,}"
                )
                report.warnings.append(message)
                log.warning("row_count_mismatch", extra={"table": spec.table_name,
                                                         "got": inserted,
                                                         "expected": spec.expected_rows})

        # Rollup last: it reads from everything above.
        summary_started = time.perf_counter()
        connection.exec_driver_sql(ACCOUNT_SUMMARY_SQL)
        summary_rows = connection.execute(
            text("SELECT COUNT(*) FROM account_summary")
        ).scalar_one()
        report.tables["account_summary"] = int(summary_rows)
        log.info(
            "summary_built",
            extra={"rows": summary_rows,
                   "seconds": round(time.perf_counter() - summary_started, 2)},
        )

    # Restore durability, then ANALYZE. The planner statistics ANALYZE collects
    # matter once generated queries start joining five tables with no thought
    # for join order -- which is exactly what a language model produces.
    _run_outside_transaction(
        engine,
        [*_bulk_load_pragmas(engine, enable=False), "ANALYZE"],
    )
    engine.dispose()

    report.duration_seconds = time.perf_counter() - started

    if not skip_verification:
        verify_warehouse(settings, report)

    log.info(
        "warehouse_ready",
        extra={"rows": report.total_rows, "seconds": round(report.duration_seconds, 1)},
    )
    return report


def _split_sql_script(script: str) -> list[str]:
    """Split a ``.sql`` file into individually executable statements."""
    statements: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(script)

    in_line_comment = False
    in_block_comment = False
    in_single_quote = False
    in_double_quote = False

    while index < length:
        char = script[index]
        nxt = script[index + 1] if index + 1 < length else ""

        if in_line_comment:
            buffer.append(char)
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            buffer.append(char)
            if char == "*" and nxt == "/":
                buffer.append(nxt)
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if in_single_quote:
            buffer.append(char)
            if char == "'":
                if nxt == "'":          # '' is a literal quote, keep going
                    buffer.append(nxt)
                    index += 2
                    continue
                in_single_quote = False
            index += 1
            continue

        if in_double_quote:
            buffer.append(char)
            if char == '"':
                in_double_quote = False
            index += 1
            continue

        # Not inside anything: this is where the interesting decisions happen.
        if char == "-" and nxt == "-":
            in_line_comment = True
            buffer.append(char)
            index += 1
            continue
        if char == "/" and nxt == "*":
            in_block_comment = True
            buffer.append(char)
            index += 1
            continue
        if char == "'":
            in_single_quote = True
            buffer.append(char)
            index += 1
            continue
        if char == '"':
            in_double_quote = True
            buffer.append(char)
            index += 1
            continue
        if char == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            index += 1
            continue

        buffer.append(char)
        index += 1

    trailing = "".join(buffer).strip()
    if trailing:
        statements.append(trailing)

    # A chunk that is nothing but comments is not a statement.
    return [s for s in statements if _has_executable_content(s)]


def _has_executable_content(statement: str) -> bool:
    """True if ``statement`` contains anything other than comments and whitespace."""
    for line in statement.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return True
    return False


def verify_warehouse(settings: Settings | None = None,
                     report: LoadReport | None = None) -> dict[str, Any]:
    """Sanity-check a built warehouse and return the findings."""
    settings = settings or get_settings()
    engine = create_engine(settings.db_url, future=True)
    findings: dict[str, Any] = {"checks": [], "warnings": []}

    def check(name: str, sql: str, expected: Any, *, hard: bool = True) -> None:
        actual = connection.execute(text(sql)).scalar()
        ok = actual == expected
        findings["checks"].append(
            {"check": name, "expected": expected, "actual": actual, "ok": ok}
        )
        if not ok:
            message = f"{name}: expected {expected!r}, got {actual!r}"
            if hard:
                raise DataError(
                    f"Warehouse verification failed -- {message}",
                    user_message="The built warehouse failed its integrity checks. "
                                 "Re-run `nl2sql data build --force` to rebuild it.",
                    details=findings,
                )
            findings["warnings"].append(message)

    try:
        with engine.connect() as connection:
            # 1. Row counts.
            for spec in TABLE_SPECS:
                check(
                    f"rowcount:{spec.table_name}",
                    f"SELECT COUNT(*) FROM {spec.table_name}",
                    spec.expected_rows,
                    hard=False,
                )
            check("rowcount:account_summary",
                  "SELECT COUNT(*) FROM account_summary", 4_500, hard=False)

            # 2. Referential integrity. SQLite only enforces FKs when asked.
            if engine.dialect.name == "sqlite":
                violations = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
                findings["checks"].append(
                    {"check": "foreign_keys", "expected": 0,
                     "actual": len(violations), "ok": not violations}
                )
                if violations:
                    raise DataError(
                        f"{len(violations)} foreign key violations in the warehouse.",
                        details={"sample": [tuple(v) for v in violations[:5]]},
                    )

            # 3. Semantic invariants -- the ones that would quietly poison answers.
            check("every account has exactly one owner",
                  """SELECT COUNT(*) FROM (
                         SELECT account_id FROM disposition
                         WHERE disposition_type = 'OWNER'
                         GROUP BY account_id HAVING COUNT(*) <> 1
                     )""", 0)
            check("no client rows lost their gender",
                  "SELECT COUNT(*) FROM client WHERE gender NOT IN ('male','female')", 0)
            check("summary balance matches the last transaction",
                  """SELECT COUNT(*) FROM account_summary s
                     JOIN (
                        SELECT account_id, balance_after,
                               ROW_NUMBER() OVER (PARTITION BY account_id
                                 ORDER BY transaction_date DESC, transaction_id DESC) rn
                        FROM bank_transaction
                     ) t ON t.account_id = s.account_id AND t.rn = 1
                     WHERE ABS(s.current_balance - t.balance_after) > 0.001""", 0)
            check("defaulted flag agrees with status code",
                  """SELECT COUNT(*) FROM loan
                     WHERE is_defaulted <> (CASE WHEN status_code IN ('B','D')
                                            THEN 1 ELSE 0 END)""", 0)

            # 4. Dates decoded into the range the dataset actually covers.
            check("transactions start in 1993",
                  "SELECT MIN(transaction_date) FROM bank_transaction", "1993-01-01",
                  hard=False)
            check("transactions end in 1998",
                  "SELECT MAX(transaction_date) FROM bank_transaction", "1998-12-31",
                  hard=False)
    finally:
        engine.dispose()

    if report is not None:
        report.warnings.extend(findings["warnings"])

    failed = [c for c in findings["checks"] if not c["ok"]]
    log.info(
        "verification_complete",
        extra={"checks": len(findings["checks"]), "failed": len(failed)},
    )
    return findings


def warehouse_exists(settings: Settings | None = None) -> bool:
    """Is there a usable warehouse on disk?"""
    settings = settings or get_settings()
    if not settings.db_file.exists():
        return False
    try:
        engine = create_engine(settings.db_url, future=True)
        with engine.connect() as connection:
            count = connection.execute(text("SELECT COUNT(*) FROM account")).scalar_one()
        engine.dispose()
        return int(count) > 0
    except Exception:
        return False
