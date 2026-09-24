"""How we decide whether a generated query was right."""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from typing import Any

import sqlglot

#: Decimal places used when comparing floats. Two is right for money, and
#: coarse enough to absorb the last-bit differences that AVG() produces
#: depending on the order rows were summed in.
FLOAT_PRECISION = 2


def normalise_cell(value: Any) -> Any:
    """Put one cell into a canonical form for comparison."""
    if value is None:
        return None
    if isinstance(value, bool):
        # Must precede the int check -- bool is a subclass of int, and SQLite
        # returns 0/1 for boolean expressions.
        return int(value)
    if isinstance(value, (int, float)):
        rounded = round(float(value), FLOAT_PRECISION)
        # Collapse -0.0 to 0.0, which otherwise compares unequal in a tuple.
        return rounded + 0.0
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def normalise_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(normalise_cell(cell) for cell in row) for row in rows]


def gold_requires_order(sql: str, dialect: str = "sqlite") -> bool:
    """Does the gold query have a top-level ORDER BY that the answer depends on?"""
    try:
        statement = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return False
    if statement is None:
        return False

    order = statement.args.get("order")
    if order is None:
        return False

    # An ORDER BY with no LIMIT on an aggregate is presentational -- the set of
    # rows is the answer, and their sequence is not. Only treat ordering as
    # part of the answer when a LIMIT makes it so ("the top 5" genuinely
    # depends on the order).
    return statement.args.get("limit") is not None


def results_match(
    gold_rows: list[tuple],
    pred_rows: list[tuple],
    *,
    order_matters: bool,
) -> bool:
    """Are these two result sets the same answer?"""
    gold, pred = normalise_rows(gold_rows), normalise_rows(pred_rows)
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    width = len(gold[0])
    if len(pred[0]) != width:
        return False

    def canonical(rows: list[tuple]) -> list[tuple]:
        return rows if order_matters else sorted(rows, key=_sort_key)

    target = canonical(gold)
    # ponytail: brute-force permutations; capped at 6 columns (720 tries), beyond that exact order only.
    orders = itertools.permutations(range(width)) if width <= 6 else [tuple(range(width))]
    return any(canonical([tuple(row[i] for i in order) for row in pred]) == target
               for order in orders)


def _sort_key(row: tuple) -> tuple:
    """Sort rows deterministically despite mixed types (None vs int) within a column."""
    return tuple((type(cell).__name__, str(cell)) for cell in row)


# ===========================================================================
# Result containers
# ===========================================================================

@dataclass
class CaseResult:
    """The outcome of evaluating one benchmark question."""

    case_id: str
    question: str
    difficulty: str
    category: str

    gold_sql: str | None = None
    predicted_sql: str = ""

    correct: bool = False
    #: The generated SQL parsed, passed the guardrails, and ran without error.
    #: True even when the answer was wrong -- these measure different things.
    executed: bool = False
    guardrail_blocked: bool = False
    refused: bool = False
    repair_attempts: int = 0

    error: str | None = None
    failure_reason: str = ""

    gold_row_count: int = 0
    predicted_row_count: int = 0

    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "difficulty": self.difficulty,
            "category": self.category,
            "gold_sql": self.gold_sql,
            "predicted_sql": self.predicted_sql,
            "correct": self.correct,
            "executed": self.executed,
            "guardrail_blocked": self.guardrail_blocked,
            "refused": self.refused,
            "repair_attempts": self.repair_attempts,
            "error": self.error,
            "failure_reason": self.failure_reason,
            "gold_row_count": self.gold_row_count,
            "predicted_row_count": self.predicted_row_count,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class EvalReport:
    """Aggregate results across a whole benchmark run."""

    provider: str = ""
    model: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    cases: list[CaseResult] = field(default_factory=list)
    started_at: str = ""
    duration_seconds: float = 0.0

    # -- headline numbers ---------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def answerable(self) -> list[CaseResult]:
        """Cases with a gold query. Refusal cases are scored separately."""
        return [c for c in self.cases if c.difficulty != "refusal"]

    @property
    def refusal_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if c.difficulty == "refusal"]

    @property
    def execution_accuracy(self) -> float:
        """The headline metric: fraction of answerable questions answered correctly."""
        cases = self.answerable
        return sum(c.correct for c in cases) / len(cases) if cases else 0.0

    @property
    def valid_sql_rate(self) -> float:
        """Fraction that produced runnable SQL, whether or not it was right."""
        cases = self.answerable
        return sum(c.executed for c in cases) / len(cases) if cases else 0.0

    @property
    def refusal_accuracy(self) -> float:
        """Fraction of unanswerable questions correctly declined."""
        cases = self.refusal_cases
        return sum(c.refused for c in cases) / len(cases) if cases else 0.0

    @property
    def guardrail_block_rate(self) -> float:
        return (
            sum(c.guardrail_blocked for c in self.cases) / self.total if self.total else 0.0
        )

    @property
    def repair_rate(self) -> float:
        """Fraction that needed at least one repair attempt."""
        return (
            sum(c.repair_attempts > 0 for c in self.cases) / self.total if self.total else 0.0
        )

    @property
    def repair_success_rate(self) -> float:
        """Of the cases that needed repair, how many ended up correct."""
        repaired = [c for c in self.cases if c.repair_attempts > 0]
        return sum(c.correct for c in repaired) / len(repaired) if repaired else 0.0

    # -- distributions ------------------------------------------------------
    @property
    def latency_p50(self) -> float:
        return self._percentile(50)

    @property
    def latency_p95(self) -> float:
        return self._percentile(95)

    def _percentile(self, pct: int) -> float:
        values = sorted(c.latency_ms for c in self.cases)
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        index = min(int(len(values) * pct / 100), len(values) - 1)
        return values[index]

    @property
    def mean_latency_ms(self) -> float:
        values = [c.latency_ms for c in self.cases]
        return statistics.mean(values) if values else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.cases)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.cases)

    def by_difficulty(self) -> dict[str, dict[str, Any]]:
        return self._breakdown(lambda c: c.difficulty)

    def by_category(self) -> dict[str, dict[str, Any]]:
        return self._breakdown(lambda c: c.category)

    def _breakdown(self, key: Any) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[CaseResult]] = {}
        for case in self.cases:
            groups.setdefault(key(case), []).append(case)

        summary: dict[str, dict[str, Any]] = {}
        for name, cases in sorted(groups.items()):
            # Refusal cases are "correct" when declined, not when executed.
            if name == "refusal" or all(c.difficulty == "refusal" for c in cases):
                correct = sum(c.refused for c in cases)
            else:
                correct = sum(c.correct for c in cases)
            summary[name] = {
                "total": len(cases),
                "correct": correct,
                "accuracy": round(correct / len(cases), 4) if cases else 0.0,
                "mean_latency_ms": round(
                    statistics.mean([c.latency_ms for c in cases]), 1
                ) if cases else 0.0,
            }
        return summary

    def failures(self) -> list[CaseResult]:
        """Everything that went wrong, for the report's failure analysis."""
        return [
            c
            for c in self.cases
            if (c.difficulty == "refusal" and not c.refused)
            or (c.difficulty != "refusal" and not c.correct)
        ]

    def failure_reasons(self) -> dict[str, int]:
        """Counts by failure mode -- which bucket to work on next."""
        counts: dict[str, int] = {}
        for case in self.failures():
            reason = case.failure_reason or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "config": self.config,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 1),
            "summary": {
                "total_cases": self.total,
                "answerable_cases": len(self.answerable),
                "execution_accuracy": round(self.execution_accuracy, 4),
                "valid_sql_rate": round(self.valid_sql_rate, 4),
                "refusal_accuracy": round(self.refusal_accuracy, 4),
                "guardrail_block_rate": round(self.guardrail_block_rate, 4),
                "repair_rate": round(self.repair_rate, 4),
                "repair_success_rate": round(self.repair_success_rate, 4),
                "latency_p50_ms": round(self.latency_p50, 1),
                "latency_p95_ms": round(self.latency_p95, 1),
                "mean_latency_ms": round(self.mean_latency_ms, 1),
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost_usd, 4),
            },
            "by_difficulty": self.by_difficulty(),
            "by_category": self.by_category(),
            "failure_reasons": self.failure_reasons(),
            "cases": [c.to_dict() for c in self.cases],
        }
