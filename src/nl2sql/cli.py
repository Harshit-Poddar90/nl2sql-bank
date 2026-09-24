"""Command-line interface."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from nl2sql import __version__
from nl2sql.config import PROJECT_ROOT, LLMProvider, get_settings
from nl2sql.exceptions import NL2SQLError
from nl2sql.logging_setup import configure_logging

app = typer.Typer(
    name="nl2sql",
    help="Ask a real banking database questions in plain English.",
    add_completion=False,
    no_args_is_help=True,
)
data_app = typer.Typer(help="Build and inspect the warehouse.", no_args_is_help=True)
app.add_typer(data_app, name="data")

console = Console()
error_console = Console(stderr=True)


def _fail(exc: Exception, verbose: bool) -> None:
    """Print a readable error and exit non-zero."""
    if verbose:
        error_console.print_exception()
    message = getattr(exc, "user_message", None) or str(exc)
    error_console.print(f"\n[bold red]Error:[/bold red] {message}\n")
    details = getattr(exc, "details", None)
    if details and verbose:
        error_console.print(details)
    raise typer.Exit(1)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging and tracebacks.")] = False,
) -> None:
    """Set up logging before any subcommand runs."""
    configure_logging(level="DEBUG" if verbose else None, force=True)
    main.verbose = verbose  # type: ignore[attr-defined]


def _verbose() -> bool:
    return getattr(main, "verbose", False)


# ===========================================================================
# data
# ===========================================================================

@data_app.command("build")
def data_build(
    force: Annotated[bool, typer.Option("--force", help="Re-download the source CSVs.")] = False,
    skip_verify: Annotated[bool, typer.Option("--skip-verify", help="Skip integrity checks.")] = False,
) -> None:
    """Download the Berka dataset and build the SQLite warehouse."""
    from nl2sql.data.etl import load_warehouse

    try:
        console.print("\n[bold]Building the warehouse[/bold] (this takes about a minute)\n")
        report = load_warehouse(force_download=force, skip_verification=skip_verify)
        console.print(report.render())
        console.print("[bold green]Done.[/bold green] Try: [cyan]nl2sql ask \"how many "
                      "accounts have more than 50000?\"[/cyan]\n")
    except Exception as exc:
        _fail(exc, _verbose())


@data_app.command("verify")
def data_verify() -> None:
    """Check the warehouse against the dataset's published row counts and invariants."""
    from nl2sql.data.etl import verify_warehouse

    try:
        findings = verify_warehouse()
        table = Table(title="Warehouse verification", show_lines=False)
        table.add_column("Check")
        table.add_column("Expected", justify="right")
        table.add_column("Actual", justify="right")
        table.add_column("", justify="center")

        for check in findings["checks"]:
            table.add_row(
                str(check["check"]),
                str(check["expected"]),
                str(check["actual"]),
                "[green]OK[/green]" if check["ok"] else "[red]FAIL[/red]",
            )
        console.print(table)

        failed = [c for c in findings["checks"] if not c["ok"]]
        if failed:
            console.print(f"\n[yellow]{len(failed)} check(s) did not match.[/yellow]\n")
            raise typer.Exit(1)
        console.print("\n[bold green]All checks passed.[/bold green]\n")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc, _verbose())


@data_app.command("info")
def data_info() -> None:
    """Show what is in the warehouse."""
    from nl2sql.catalog.catalog import build_catalog

    try:
        catalog = build_catalog()
        table = Table(title="Warehouse contents")
        table.add_column("Table")
        table.add_column("Kind")
        table.add_column("Rows", justify="right")
        table.add_column("Cols", justify="right")
        table.add_column("Description", overflow="fold", max_width=52)

        for name in catalog.table_names:
            info = catalog.tables[name]
            table.add_row(
                name,
                info.kind,
                f"{info.row_count:,}",
                str(len(info.columns)),
                info.description[:100],
            )
        console.print(table)
        console.print(
            f"\n{sum(t.row_count for t in catalog.tables.values() if t.kind == 'table'):,} "
            f"rows across {len(catalog.base_table_names)} base tables.\n"
        )
    except Exception as exc:
        _fail(exc, _verbose())


# ===========================================================================
# ask
# ===========================================================================

@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question, in plain English.")],
    show_sql: Annotated[bool, typer.Option("--sql/--no-sql", help="Show the generated query.")] = True,
    show_trace: Annotated[bool, typer.Option("--trace", help="Show retrieval and repair details.")] = False,
    explain: Annotated[bool, typer.Option("--explain", help="Show the database query plan.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON to stdout.")] = False,
    offline: Annotated[bool, typer.Option("--offline", help="Use the rule-based generator, no API key needed.")] = False,
    max_rows: Annotated[int, typer.Option("--rows", help="Rows to display.")] = 20,
) -> None:
    """Ask a question and get an answer backed by SQL that actually ran."""
    from nl2sql.pipeline import Pipeline

    try:
        settings = get_settings()
        provider = LLMProvider.STUB if offline else None
        client = None
        if provider is not None:
            from nl2sql.llm.factory import create_llm_client

            client = create_llm_client(settings, provider=provider)

        pipeline = Pipeline.build(settings, llm=client, fallback_to_stub=False)
        result = pipeline.ask(question)

        if as_json:
            # Logs go to stderr, so stdout stays clean for piping into jq.
            print(json.dumps(result.to_dict(), indent=2, default=str))
            raise typer.Exit(0 if result.success else 1)

        console.print()
        if result.refused:
            console.print(Panel(result.answer, title="Cannot answer",
                                border_style="yellow"))
        elif not result.success:
            console.print(Panel(result.error or "Unknown error", title="Failed",
                                border_style="red"))
            if result.sql:
                console.print(Syntax(result.sql, "sql", theme="ansi_dark"))
            raise typer.Exit(1)
        else:
            console.print(Panel(result.answer, title="Answer", border_style="green"))

        if show_sql and result.sql:
            console.print()
            console.print(Syntax(result.sql, "sql", theme="ansi_dark",
                                 line_numbers=False, word_wrap=True))

        if result.rows:
            console.print()
            table = Table(show_header=True, header_style="bold cyan")
            for column in result.columns:
                table.add_column(column)
            for row in result.rows[:max_rows]:
                table.add_row(*["NULL" if v is None else str(v) for v in row])
            console.print(table)
            if result.row_count > max_rows:
                console.print(f"[dim]... {result.row_count - max_rows:,} more rows[/dim]")

        if explain and result.sql:
            console.print("\n[bold]Query plan[/bold]")
            for line in pipeline.executor.explain(result.sql):
                console.print(f"  [dim]{line}[/dim]")

        if show_trace:
            console.print("\n[bold]Trace[/bold]")
            console.print(f"  tables retrieved : {', '.join(result.tables_used)}")
            console.print(f"  repair attempts  : {result.repair_attempts}")
            for attempt in result.attempts:
                marker = "OK " if attempt.outcome == "success" else "ERR"
                console.print(f"    [{marker}] attempt {attempt.attempt}: {attempt.outcome}"
                              + (f" -- {attempt.error[:90]}" if attempt.error else ""))
            for warning in result.warnings:
                console.print(f"  [yellow]warning[/yellow]: {warning}")

        timings = (
            f"retrieval {result.retrieval_ms:.0f}ms | "
            f"generation {result.generation_ms:.0f}ms | "
            f"validation {result.validation_ms:.0f}ms | "
            f"execution {result.execution_ms:.0f}ms | "
            f"total {result.total_ms:.0f}ms"
        )
        console.print(f"\n[dim]{timings}[/dim]")
        if result.input_tokens or result.output_tokens:
            console.print(
                f"[dim]{result.model} | {result.input_tokens:,} in / "
                f"{result.output_tokens:,} out | ${result.cost_usd:.5f}[/dim]"
            )
        console.print()

    except typer.Exit:
        raise
    except NL2SQLError as exc:
        _fail(exc, _verbose())
    except Exception as exc:
        _fail(exc, _verbose())


# ===========================================================================
# schema / validate
# ===========================================================================

@app.command()
def schema(
    tables: Annotated[list[str] | None, typer.Argument(help="Tables to show. Omit for all.")] = None,
    question: Annotated[str, typer.Option("--for", help="Show only what the retriever would pick for this question.")] = "",
) -> None:
    """Print the schema exactly as the model receives it."""
    from nl2sql.catalog.catalog import build_catalog

    try:
        catalog = build_catalog()

        if question:
            from nl2sql.retrieval.schema_linker import SchemaLinker

            linker = SchemaLinker(catalog)
            link = linker.link(question)
            console.print(f"\n[bold]Retrieved for:[/bold] {question}")
            console.print(f"[dim]tables: {', '.join(link.tables)} | "
                          f"{link.column_count} columns | {link.latency_ms:.1f} ms[/dim]\n")
            rendered = catalog.render_schema(link.tables, link.columns)
            value_hints = link.render_value_hints()
            if value_hints:
                rendered += "\n\n" + value_hints
        else:
            rendered = catalog.render_schema(list(tables) if tables else None)

        console.print(Syntax(rendered, "sql", theme="ansi_dark", word_wrap=True))
    except Exception as exc:
        _fail(exc, _verbose())


@app.command()
def validate(
    sql: Annotated[str, typer.Argument(help="SQL to check.")],
    execute: Annotated[bool, typer.Option("--execute", help="Also run it if it passes.")] = False,
) -> None:
    """Run SQL past the guardrails without executing it."""
    from nl2sql.catalog.catalog import build_catalog
    from nl2sql.guardrails.validator import SQLValidator

    try:
        catalog = build_catalog()
        result = SQLValidator(catalog).validate(sql)

        console.print()
        if result.is_valid:
            console.print(Panel("[bold green]ACCEPTED[/bold green]", border_style="green"))
            console.print("\n[bold]Rewritten query (this is what would run):[/bold]")
            console.print(Syntax(result.sql, "sql", theme="ansi_dark"))
            console.print(f"\n  tables      : {', '.join(result.tables_referenced)}")
            console.print(f"  row limit   : {result.limit_applied}"
                          + (" [yellow](injected)[/yellow]" if result.limit_was_injected else ""))
        else:
            console.print(Panel("[bold red]REJECTED[/bold red]", border_style="red"))
            for violation in result.violations:
                console.print(f"  [red]x[/red] {violation}")

        for warning in result.warnings:
            console.print(f"  [yellow]![/yellow] {warning}")
        console.print(f"\n[dim]validated in {result.latency_ms:.2f} ms[/dim]\n")

        if execute and result.is_valid:
            from nl2sql.execution.executor import QueryExecutor

            query_result = QueryExecutor().execute(result.sql)
            console.print(query_result.to_markdown())
            console.print()

        raise typer.Exit(0 if result.is_valid else 1)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc, _verbose())


# ===========================================================================
# eval
# ===========================================================================

@app.command("eval")
def evaluate(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Only the first N cases.")] = 0,
    difficulty: Annotated[str, typer.Option("--difficulty", "-d", help="easy | medium | hard | refusal")] = "",
    category: Annotated[str, typer.Option("--category", "-c", help="Filter by category.")] = "",
    provider: Annotated[str, typer.Option("--provider", "-p", help="Override the provider.")] = "",
    no_retrieval: Annotated[bool, typer.Option("--no-retrieval", help="Ablation: send the whole schema.")] = False,
    no_few_shot: Annotated[bool, typer.Option("--no-few-shot", help="Ablation: no worked examples.")] = False,
    no_repair: Annotated[bool, typer.Option("--no-repair", help="Ablation: disable the repair loop.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Summary only.")] = False,
) -> None:
    """Run the benchmark and write a report."""
    from nl2sql.evaluation.runner import print_summary, run_benchmark

    try:
        settings = get_settings()

        # Ablations mutate a copy of the settings for this run only.
        overrides: dict[str, object] = {}
        labels: list[str] = []
        if no_retrieval:
            overrides["retrieval_enabled"] = False
            labels.append("no-retrieval")
        if no_few_shot:
            overrides["few_shot_count"] = 0
            labels.append("no-few-shot")
        if no_repair:
            overrides["max_repair_attempts"] = 0
            labels.append("no-repair")

        if overrides:
            settings = settings.model_copy(update=overrides)
            console.print(f"[yellow]Ablation:[/yellow] {', '.join(labels)}")

        report = run_benchmark(
            settings,
            limit=limit or None,
            difficulty=difficulty or None,
            category=category or None,
            provider=LLMProvider(provider) if provider else None,
            progress=not quiet,
            label="-".join(labels),
        )
        print_summary(report)
    except Exception as exc:
        _fail(exc, _verbose())


# ===========================================================================
# history / config / serve / ui
# ===========================================================================

@app.command()
def history(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 15,
    failures_only: Annotated[bool, typer.Option("--failures", help="Only failed requests.")] = False,
) -> None:
    """Show recent questions from the audit log."""
    from nl2sql.observability.query_log import QueryLog

    try:
        log = QueryLog()
        entries = log.failures(limit) if failures_only else log.recent(limit)

        if not entries:
            console.print("\n[dim]Nothing recorded yet.[/dim]\n")
            return

        table = Table(title="Recent questions")
        table.add_column("When", style="dim")
        table.add_column("Question", overflow="fold", max_width=42)
        table.add_column("", justify="center")
        table.add_column("Rows", justify="right")
        table.add_column("Fix", justify="right")
        table.add_column("ms", justify="right")

        for entry in entries:
            status = "[green]ok[/green]" if entry["success"] else (
                "[yellow]refused[/yellow]" if entry["refused"] else "[red]fail[/red]"
            )
            table.add_row(
                str(entry["created_at"])[5:16],
                str(entry["question"]),
                status,
                str(entry["row_count"] if entry["row_count"] is not None else "-"),
                str(entry["repair_attempts"]),
                f"{entry['total_ms']:.0f}" if entry["total_ms"] else "-",
            )
        console.print(table)

        stats = log.stats()
        console.print(
            f"\n[dim]{stats['total_queries']} total | "
            f"success rate {(stats['success_rate'] or 0) * 100:.0f}% | "
            f"{stats['total_tokens']:,} tokens | "
            f"${stats['total_cost_usd']:.4f}[/dim]\n"
        )
    except Exception as exc:
        _fail(exc, _verbose())


@app.command("config")
def show_config() -> None:
    """Show the active configuration, with secrets redacted."""
    from nl2sql.data.etl import warehouse_exists
    from nl2sql.llm.factory import available_providers

    settings = get_settings()
    table = Table(title=f"nl2sql-bank {__version__}")
    table.add_column("Setting")
    table.add_column("Value", overflow="fold")

    for key, value in settings.safe_summary().items():
        table.add_row(key, str(value))
    table.add_row("warehouse built", "yes" if warehouse_exists(settings) else "[red]no[/red]")
    console.print(table)

    providers = available_providers(settings)
    console.print("\n[bold]Providers[/bold]")
    for name, ready in providers.items():
        mark = "[green]ready[/green]" if ready else "[dim]no key[/dim]"
        console.print(f"  {name:<12} {mark}")

    if not settings.active_api_key and settings.requires_api_key:
        console.print(
            f"\n[yellow]No API key set for '{settings.llm_provider.value}'.[/yellow]\n"
            f"Add one to .env:  NL2SQL_{settings.llm_provider.value.upper()}_API_KEY=...\n"
            f"Or run offline:   nl2sql ask \"...\" --offline\n"
        )
    console.print()


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "",
    port: Annotated[int, typer.Option("--port")] = 0,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes.")] = False,
) -> None:
    """Start the HTTP API."""
    import uvicorn

    settings = get_settings()
    host = host or settings.api_host
    port = port or settings.api_port

    console.print(f"\n[bold green]API[/bold green]  http://{host}:{port}")
    console.print(f"[dim]Interactive docs at http://{host}:{port}/docs[/dim]\n")
    uvicorn.run("nl2sql.api.main:app", host=host, port=port, reload=reload,
                log_level=settings.log_level.lower())


@app.command()
def ui(
    port: Annotated[int, typer.Option("--port")] = 8501,
) -> None:
    """Start the Streamlit demo."""
    import subprocess
    from pathlib import Path

    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    if not app_path.exists():
        error_console.print(f"[red]Not found:[/red] {app_path}")
        raise typer.Exit(1)

    console.print(f"\n[bold green]UI[/bold green]  http://localhost:{port}\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path),
             "--server.port", str(port)],
            check=True,
            cwd=str(Path(app_path).parent.parent),
        )
    except FileNotFoundError:
        error_console.print("[red]Streamlit is not installed.[/red] "
                            "Install it with: pip install 'nl2sql-bank[ui]'")
        raise typer.Exit(1) from None
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(exc.returncode) from exc


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"nl2sql-bank {__version__}")


if __name__ == "__main__":
    app()
