"""Streamlit demo UI."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the package importable when run via `streamlit run` from the repo root,
# which does not put `src/` on the path.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nl2sql import __version__  # noqa: E402
from nl2sql.config import LLMProvider, get_settings  # noqa: E402
from nl2sql.data.etl import warehouse_exists  # noqa: E402
from nl2sql.llm.factory import create_llm_client  # noqa: E402
from nl2sql.observability.query_log import QueryLog  # noqa: E402
from nl2sql.pipeline import Pipeline  # noqa: E402

st.set_page_config(
    page_title="nl2sql-bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "How many accounts have more than 50,000 in them?",
    "Which region has the highest loan default rate?",
    "What is the average balance of accounts owned by women?",
    "How many gold cards were issued in 1996?",
    "Show me the 10 accounts with the largest balances",
    "How many clients older than 60 have a defaulted loan?",
    "Compare the average balance of accounts with a loan against those without",
    "What are the names and email addresses of our clients?",
]

ATTACK_EXAMPLES = [
    "SELECT 1; DROP TABLE loan",
    "DELETE FROM account_summary WHERE 1=1",
    "SELECT name, email FROM clients",
    "SELECT load_extension('/tmp/evil.so')",
    "SELECT * FROM bank_transaction, client, account, loan",
    "SELECT account_id FROM account_summary LIMIT 999999",
]


@st.cache_resource(show_spinner="Loading the warehouse and embedding model...")
def load_pipeline(provider_name: str) -> Pipeline:
    """Build the pipeline once and reuse it across reruns."""
    settings = get_settings()
    client = None
    if provider_name:
        client = create_llm_client(settings, provider=LLMProvider(provider_name),
                                   fallback_to_stub=True)
    return Pipeline.build(settings, llm=client, fallback_to_stub=True)


def main() -> None:
    settings = get_settings()

    # -- guard: nothing works without the warehouse ------------------------
    if not warehouse_exists(settings):
        st.error("The warehouse has not been built yet.")
        st.code("nl2sql data build", language="bash")
        st.caption("Downloads 68 MB of source CSVs and builds a 1,084,180-row "
                   "SQLite database. Takes about a minute.")
        st.stop()

    # -- sidebar ------------------------------------------------------------
    with st.sidebar:
        st.title("🏦 nl2sql-bank")
        st.caption(f"v{__version__} — natural language over a real banking warehouse")

        has_key = bool(settings.active_api_key)
        default_provider = settings.llm_provider.value if has_key else "stub"

        provider = st.selectbox(
            "Model provider",
            options=[p.value for p in LLMProvider],
            index=[p.value for p in LLMProvider].index(default_provider),
            help="`stub` is the offline rule-based baseline. It needs no API key "
                 "and answers simple questions only.",
        )

        if provider != "stub" and not has_key:
            st.warning(
                f"No API key for **{provider}**. Add "
                f"`NL2SQL_{provider.upper()}_API_KEY` to your `.env` file, "
                f"or use `stub` to try it offline."
            )

        st.divider()
        st.subheader("The data")
        st.markdown(
            """
**PKDD'99 Berka** - a real Czech bank, 1993-1998.

| | |
|---|---|
| Transactions | 1,056,320 |
| Accounts | 4,500 |
| Clients | 5,369 |
| Loans | 682 |
| Districts | 77 |

Money is Czech koruna (CZK).
            """
        )

        st.divider()
        with st.expander("Pipeline stages"):
            st.markdown(
                """
1. **Schema linking** — BM25 + embeddings + value index, fused with RRF
2. **Generation** — retrieved schema + nearest-neighbour examples → LLM
3. **Guardrails** — AST validation, identifier allowlist, forced LIMIT
4. **Execution** — read-only connection, wall-clock timeout, row cap
5. **Repair** — on failure, feed the error back and retry (bounded)
6. **Answer** — result set → plain English
                """
            )

    pipeline = load_pipeline(provider)

    # -- tabs ---------------------------------------------------------------
    ask_tab, guardrail_tab, schema_tab, history_tab = st.tabs(
        ["Ask", "Guardrails", "Schema", "History"]
    )

    with ask_tab:
        render_ask(pipeline)
    with guardrail_tab:
        render_guardrails(pipeline)
    with schema_tab:
        render_schema(pipeline)
    with history_tab:
        render_history(settings)


def render_ask(pipeline: Pipeline) -> None:
    """The main question-answering view."""
    st.header("Ask a question")

    st.caption("Try one of these:")
    columns = st.columns(4)
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 4].button(example, key=f"ex{index}", width="stretch"):
            st.session_state["question"] = example

    question = st.text_input(
        "Your question",
        value=st.session_state.get("question", ""),
        placeholder="How many accounts have more than 50,000 in them?",
        label_visibility="collapsed",
    )

    if not st.button("Ask", type="primary") and not question:
        return
    if not question:
        st.info("Type a question, or click one of the examples above.")
        return

    with st.spinner("Thinking..."):
        result = pipeline.ask(question)

    # -- the answer
    if result.refused:
        st.warning(f"**Cannot answer.** {result.answer}")
        st.caption(
            "This is the system declining rather than inventing a schema. "
            "Refusing correctly is measured in the benchmark too."
        )
    elif not result.success:
        st.error(f"**Failed.** {result.error}")
    else:
        st.success(result.answer)

    # -- the query
    if result.sql:
        st.subheader("The query that ran")
        st.code(result.sql, language="sql")

    # -- the rows
    if result.rows:
        st.subheader(f"Results ({result.row_count:,} row{'s' if result.row_count != 1 else ''})")
        frame = pd.DataFrame(result.rows, columns=result.columns)
        st.dataframe(frame, width="stretch", hide_index=True)
        if result.truncated:
            st.caption("Capped at the row limit — this is a partial result.")

    # -- the working
    metrics = st.columns(5)
    metrics[0].metric("Total", f"{result.total_ms:.0f} ms")
    metrics[1].metric("Retrieval", f"{result.retrieval_ms:.0f} ms")
    metrics[2].metric("Generation", f"{result.generation_ms:.0f} ms")
    metrics[3].metric("Execution", f"{result.execution_ms:.0f} ms")
    metrics[4].metric("Repairs", result.repair_attempts)

    with st.expander("How this answer was produced"):
        left, right = st.columns(2)

        with left:
            st.markdown("**Tables the retriever selected**")
            for table in result.tables_used:
                st.markdown(f"- `{table}`")

        with right:
            st.markdown("**Model usage**")
            st.markdown(
                f"- provider: `{result.provider or 'n/a'}`\n"
                f"- model: `{result.model or 'n/a'}`\n"
                f"- tokens: {result.input_tokens:,} in / {result.output_tokens:,} out\n"
                f"- cost: ${result.cost_usd:.5f}"
            )

        if len(result.attempts) > 1:
            st.markdown("**Repair trace**")
            for attempt in result.attempts:
                icon = "✅" if attempt.outcome == "success" else "❌"
                st.markdown(f"{icon} **Attempt {attempt.attempt + 1}** — `{attempt.outcome}`")
                if attempt.error:
                    st.caption(attempt.error[:300])
                st.code(attempt.sql, language="sql")

        if result.warnings:
            st.markdown("**Warnings**")
            for warning in result.warnings:
                st.caption(f"⚠️ {warning}")


def render_guardrails(pipeline: Pipeline) -> None:
    """A playground for the safety layer. Try to get something dangerous through."""
    st.header("Guardrail playground")
    st.markdown(
        "Every generated query is parsed into an AST and checked before it "
        "reaches the database. Paste anything you like below — nothing typed "
        "here is executed."
    )

    st.caption("Try one of these:")
    columns = st.columns(3)
    for index, attack in enumerate(ATTACK_EXAMPLES):
        if columns[index % 3].button(attack[:36], key=f"atk{index}", width="stretch"):
            st.session_state["attack_sql"] = attack

    sql = st.text_area(
        "SQL to validate",
        value=st.session_state.get("attack_sql", "SELECT 1; DROP TABLE loan"),
        height=110,
    )

    if not sql.strip():
        return

    result = pipeline.validator.validate(sql)

    if result.is_valid:
        st.success("**ACCEPTED** — this query is safe to run.")
        st.markdown("**Rewritten query (note the injected row limit):**")
        st.code(result.sql, language="sql")
        left, right = st.columns(2)
        left.metric("Row limit", result.limit_applied or "none")
        right.metric("Limit injected", "yes" if result.limit_was_injected else "no")
        if result.tables_referenced:
            st.caption("Tables referenced: " + ", ".join(f"`{t}`" for t in result.tables_referenced))
    else:
        st.error("**REJECTED** — blocked before reaching the database.")
        for violation in result.violations:
            st.markdown(f"- {violation}")

    for warning in result.warnings:
        st.warning(warning)
    st.caption(f"Validated in {result.latency_ms:.2f} ms")


def render_schema(pipeline: Pipeline) -> None:
    """Schema browser, plus a live view of what the retriever picks."""
    st.header("Schema")
    catalog = pipeline.catalog

    st.subheader("Retrieval preview")
    st.caption(
        "Type a question to see which slice of the schema the retriever selects "
        "and puts in the prompt. This is what makes the approach scale past a "
        "schema that fits in a context window."
    )
    probe = st.text_input("Question", placeholder="which region has the most defaulted loans?")

    if probe:
        link = pipeline.generator.linker.link(probe)

        left, right = st.columns(2)
        left.metric("Tables selected", f"{len(link.tables)} of {len(catalog.tables)}")
        right.metric("Retrieval latency", f"{link.latency_ms:.1f} ms")

        if link.value_hits:
            st.markdown("**Values matched directly in the data:**")
            for hit in link.value_hits:
                st.markdown(f"- `{hit.phrase}` → `{hit.qualified_column}` (value: `{hit.value}`)")

        st.code(catalog.render_schema(link.tables, link.columns), language="sql")
    else:
        st.subheader("Tables")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "table": name,
                        "kind": info.kind,
                        "rows": info.row_count,
                        "columns": len(info.columns),
                        "description": info.description[:90],
                    }
                    for name, info in sorted(catalog.tables.items())
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        chosen = st.selectbox("Show full definition for", sorted(catalog.tables))
        if chosen:
            st.code(catalog.tables[chosen].render(), language="sql")


def render_history(settings) -> None:  # type: ignore[no-untyped-def]
    """The audit log."""
    st.header("Query history")
    st.caption(
        "Every request is recorded: the question, the SQL, timings, tokens and "
        "the outcome. This is the audit trail that makes the system debuggable "
        "in production."
    )

    query_log = QueryLog(settings=settings)
    stats = query_log.stats()

    if not stats.get("total_queries"):
        st.info("No questions recorded yet. Ask something on the **Ask** tab.")
        return

    columns = st.columns(5)
    columns[0].metric("Total", f"{stats['total_queries']:,}")
    columns[1].metric("Success rate", f"{(stats.get('success_rate') or 0) * 100:.0f}%")
    columns[2].metric("Needed repair", stats.get("needed_repair", 0))
    columns[3].metric("Blocked", stats.get("guardrail_blocked", 0))
    columns[4].metric("Cost", f"${stats.get('total_cost_usd', 0):.4f}")

    only_failures = st.checkbox("Failures only")
    entries = query_log.failures(50) if only_failures else query_log.recent(50)

    if not entries:
        st.info("Nothing to show.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "when": e["created_at"][5:16],
                    "question": e["question"],
                    "ok": "yes" if e["success"] else ("refused" if e["refused"] else "no"),
                    "rows": e["row_count"],
                    "repairs": e["repair_attempts"],
                    "ms": round(e["total_ms"]) if e["total_ms"] else None,
                }
                for e in entries
            ]
        ),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
