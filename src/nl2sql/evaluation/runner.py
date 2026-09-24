"""Running the benchmark and writing the report."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nl2sql.config import LLMProvider, Settings, get_settings
from nl2sql.evaluation.metrics import (
    CaseResult,
    EvalReport,
    gold_requires_order,
    results_match,
)
from nl2sql.execution.executor import QueryExecutor
from nl2sql.llm.factory import create_llm_client
from nl2sql.logging_setup import get_logger
from nl2sql.pipeline import AskResult, Pipeline

log = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkCase:
    """One gold question."""

    id: str
    question: str
    gold_sql: str | None
    difficulty: str
    category: str
    note: str = ""

    @property
    def is_refusal(self) -> bool:
        return self.gold_sql is None


def load_benchmark(path: Path | None = None, settings: Settings | None = None) -> list[BenchmarkCase]:
    """Read benchmark.jsonl."""
    settings = settings or get_settings()
    path = path or settings.benchmark_path

    if not path.exists():
        raise FileNotFoundError(f"No benchmark file at {path}")

    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {line_number} is not valid JSON: {exc}") from exc

        cases.append(
            BenchmarkCase(
                id=raw["id"],
                question=raw["question"],
                gold_sql=raw.get("gold_sql"),
                difficulty=raw.get("difficulty", "unknown"),
                category=raw.get("category", "unknown"),
                note=raw.get("note", ""),
            )
        )
    return cases


def classify_failure(case: BenchmarkCase, result: AskResult, executed: bool) -> str:
    """Bucket a failure into a named mode."""
    if case.is_refusal and not result.refused:
        return "missed_refusal"
    if not case.is_refusal and result.refused:
        return "refused_answerable"
    if result.error_type == "QueryTimeoutError":
        return "timeout"
    if result.guardrail_blocked and not executed:
        return "guardrail_blocked"
    if result.error_type in {"SQLGenerationError", "LLMResponseError"}:
        return "generation_error"
    if not executed:
        return "execution_error"
    return "wrong_result"


class BenchmarkRunner:
    """Executes the benchmark against a pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        settings: Settings | None = None,
        executor: QueryExecutor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline
        # A separate executor for the gold queries, so gold execution never
        # appears in the pipeline's own timings or audit log.
        self.gold_executor = executor or QueryExecutor(self.settings)

    def run(
        self,
        cases: list[BenchmarkCase],
        *,
        progress: bool = True,
    ) -> EvalReport:
        """Evaluate every case and return the report."""
        started = time.perf_counter()
        report = EvalReport(
            provider=self.settings.llm_provider.value,
            model=self.settings.resolved_model,
            started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config={
                "retrieval_enabled": self.settings.retrieval_enabled,
                "embeddings_enabled": self.settings.embeddings_enabled,
                "few_shot_count": self.settings.few_shot_count,
                "max_repair_attempts": self.settings.max_repair_attempts,
                "temperature": self.settings.llm_temperature,
                "max_result_rows": self.settings.max_result_rows,
            },
        )

        for index, case in enumerate(cases, 1):
            case_result = self._run_case(case)
            report.cases.append(case_result)

            if progress:
                mark = "PASS" if self._passed(case, case_result) else "FAIL"
                print(
                    f"  [{index:>3}/{len(cases)}] {mark}  {case.id:<12} "
                    f"{case.question[:56]}",
                    flush=True,
                )

        report.duration_seconds = time.perf_counter() - started
        log.info(
            "benchmark_complete",
            extra={
                "cases": len(cases),
                "accuracy": round(report.execution_accuracy, 4),
                "seconds": round(report.duration_seconds, 1),
            },
        )
        return report

    @staticmethod
    def _passed(case: BenchmarkCase, result: CaseResult) -> bool:
        return result.refused if case.is_refusal else result.correct

    def _run_case(self, case: BenchmarkCase) -> CaseResult:
        """Ask one question and score the answer."""
        outcome = CaseResult(
            case_id=case.id,
            question=case.question,
            difficulty=case.difficulty,
            category=case.category,
            gold_sql=case.gold_sql,
        )

        result = self.pipeline.ask(case.question)

        outcome.predicted_sql = result.sql
        outcome.refused = result.refused
        outcome.repair_attempts = result.repair_attempts
        outcome.guardrail_blocked = result.guardrail_blocked
        outcome.error = result.error
        outcome.latency_ms = result.total_ms
        outcome.input_tokens = result.input_tokens
        outcome.output_tokens = result.output_tokens
        outcome.cost_usd = result.cost_usd
        outcome.predicted_row_count = result.row_count
        outcome.executed = result.success and not result.refused

        # -- refusal cases: correct exactly when the system declined.
        if case.is_refusal:
            outcome.correct = result.refused
            if not outcome.correct:
                outcome.failure_reason = classify_failure(case, result, outcome.executed)
            return outcome

        if not outcome.executed:
            outcome.failure_reason = classify_failure(case, result, outcome.executed)
            return outcome

        # -- answerable: compare against the gold result set.
        try:
            gold = self.gold_executor.execute(
                case.gold_sql or "",
                # Gold queries are trusted and hand-written, so they get a
                # generous row cap. Capping them at the serving limit could
                # make a correct prediction compare unequal purely because the
                # two were truncated at different points.
                max_rows=100_000,
            )
        except Exception as exc:
            outcome.failure_reason = "gold_query_error"
            outcome.error = f"Gold query failed: {exc}"
            log.error("gold_query_failed", extra={"case": case.id, "error": str(exc)})
            return outcome

        outcome.gold_row_count = gold.row_count
        outcome.correct = results_match(
            gold.rows,
            result.rows,
            order_matters=gold_requires_order(case.gold_sql or ""),
        )

        if not outcome.correct:
            outcome.failure_reason = "wrong_result"

        return outcome


# ===========================================================================
# Reporting
# ===========================================================================

def render_markdown(report: EvalReport) -> str:
    """Format the report as markdown, ready to paste into a README."""
    summary = report.to_dict()["summary"]

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    lines: list[str] = [
        "# Benchmark report",
        "",
        f"**Model:** `{report.provider}:{report.model}`  ",
        f"**Run at:** {report.started_at}  ",
        f"**Duration:** {report.duration_seconds:.1f}s  ",
        f"**Cases:** {report.total} "
        f"({len(report.answerable)} answerable, {len(report.refusal_cases)} unanswerable)",
        "",
        "## Headline",
        "",
        "| Metric | Value | What it means |",
        "|---|---|---|",
        f"| **Execution accuracy** | **{pct(report.execution_accuracy)}** | "
        f"Answered correctly, verified by comparing result sets |",
        f"| Valid SQL rate | {pct(report.valid_sql_rate)} | "
        f"Produced runnable SQL, right or wrong |",
        f"| Refusal accuracy | {pct(report.refusal_accuracy)} | "
        f"Correctly declined the unanswerable questions |",
        f"| Guardrail block rate | {pct(report.guardrail_block_rate)} | "
        f"Queries the safety layer rejected |",
        f"| Needed repair | {pct(report.repair_rate)} | "
        f"Required at least one self-correction |",
        f"| Repair success rate | {pct(report.repair_success_rate)} | "
        f"Of those, how many ended up correct |",
        "",
        "## Performance and cost",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Median latency | {summary['latency_p50_ms']:,.0f} ms |",
        f"| p95 latency | {summary['latency_p95_ms']:,.0f} ms |",
        f"| Mean latency | {summary['mean_latency_ms']:,.0f} ms |",
        f"| Total tokens | {summary['total_tokens']:,} |",
        f"| Total cost | ${summary['total_cost_usd']:.4f} |",
        f"| Cost per question | ${summary['total_cost_usd'] / max(report.total, 1):.5f} |",
        "",
        "## By difficulty",
        "",
        "| Difficulty | Cases | Correct | Accuracy | Mean latency |",
        "|---|---|---|---|---|",
    ]

    for name, stats in report.by_difficulty().items():
        lines.append(
            f"| {name} | {stats['total']} | {stats['correct']} | "
            f"{stats['accuracy'] * 100:.1f}% | {stats['mean_latency_ms']:,.0f} ms |"
        )

    lines += ["", "## By category", "", "| Category | Cases | Correct | Accuracy |", "|---|---|---|---|"]
    for name, stats in report.by_category().items():
        lines.append(
            f"| {name} | {stats['total']} | {stats['correct']} | "
            f"{stats['accuracy'] * 100:.1f}% |"
        )

    reasons = report.failure_reasons()
    if reasons:
        lines += ["", "## Failure modes", "", "| Reason | Count |", "|---|---|"]
        lines += [f"| {reason} | {count} |" for reason, count in reasons.items()]

    failures = report.failures()
    if failures:
        lines += ["", "## Failed cases", ""]
        for case in failures[:25]:
            lines += [
                f"### `{case.case_id}` -- {case.question}",
                "",
                f"*Reason:* `{case.failure_reason}`"
                + (f" -- {case.error}" if case.error else ""),
                "",
                "```sql",
                f"-- gold:\n{case.gold_sql or '(should have refused)'}",
                "",
                f"-- predicted:\n{case.predicted_sql or '(none)'}",
                "```",
                "",
            ]
        if len(failures) > 25:
            lines.append(f"_... and {len(failures) - 25} more failures (see the JSON)._")

    lines += [
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(report.config, indent=2),
        "```",
        "",
    ]

    return "\n".join(lines)


def save_report(report: EvalReport, settings: Settings | None = None,
                label: str = "") -> tuple[Path, Path]:
    """Write the markdown and JSON reports. Returns both paths."""
    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{label}" if label else ""
    base = f"eval-{report.provider}-{stamp}{suffix}"

    markdown_path = settings.reports_dir / f"{base}.md"
    json_path = settings.reports_dir / f"{base}.json"

    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    # `latest.md` is the one committed to the repo and linked from the README,
    # so the numbers there are always the most recent real run.
    (settings.reports_dir / "latest.md").write_text(
        render_markdown(report), encoding="utf-8"
    )

    return markdown_path, json_path


def run_benchmark(
    settings: Settings | None = None,
    *,
    limit: int | None = None,
    difficulty: str | None = None,
    category: str | None = None,
    provider: LLMProvider | None = None,
    progress: bool = True,
    save: bool = True,
    label: str = "",
) -> EvalReport:
    """Run the benchmark end to end. The entry point the CLI calls."""
    settings = settings or get_settings()
    cases = load_benchmark(settings=settings)

    if difficulty:
        cases = [c for c in cases if c.difficulty == difficulty]
    if category:
        cases = [c for c in cases if c.category == category]
    if limit:
        cases = cases[:limit]

    if not cases:
        raise ValueError("No benchmark cases matched the given filters.")

    client = create_llm_client(settings, provider=provider, fallback_to_stub=False)
    pipeline = Pipeline.build(settings, llm=client, enable_logging=False)

    if progress:
        print(
            f"\nRunning {len(cases)} cases against "
            f"{client.provider}:{client.model}\n"
        )

    try:
        report = BenchmarkRunner(pipeline, settings).run(cases, progress=progress)
        # Record the provider actually used, which may differ from the config
        # when `provider=` was passed.
        report.provider = client.provider
        report.model = client.model
    finally:
        pipeline.close()

    if save:
        markdown_path, json_path = save_report(report, settings, label=label)
        if progress:
            print(f"\nReport written to {markdown_path}")
            print(f"Raw results:     {json_path}")

    return report


def print_summary(report: EvalReport) -> None:
    """A compact terminal summary."""
    print()
    print("=" * 62)
    print(f"  {report.provider}:{report.model}")
    print("=" * 62)
    print(f"  Execution accuracy   {report.execution_accuracy * 100:>6.1f}%   "
          f"({sum(c.correct for c in report.answerable)}/{len(report.answerable)})")
    print(f"  Valid SQL rate       {report.valid_sql_rate * 100:>6.1f}%")
    print(f"  Refusal accuracy     {report.refusal_accuracy * 100:>6.1f}%   "
          f"({sum(c.refused for c in report.refusal_cases)}/{len(report.refusal_cases)})")
    print(f"  Needed repair        {report.repair_rate * 100:>6.1f}%")
    print(f"  Median latency       {report.latency_p50:>6.0f} ms")
    print(f"  Total cost           ${report.total_cost_usd:>7.4f}")
    print("-" * 62)
    for name, stats in report.by_difficulty().items():
        print(f"  {name:<10} {stats['correct']:>3}/{stats['total']:<3} "
              f"{stats['accuracy'] * 100:>6.1f}%")
    reasons = report.failure_reasons()
    if reasons:
        print("-" * 62)
        print("  Failure modes:")
        for reason, count in reasons.items():
            print(f"    {reason:<22} {count}")
    print("=" * 62)
    print()
