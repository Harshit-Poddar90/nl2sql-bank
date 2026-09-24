"""Prompt construction."""

from __future__ import annotations

from nl2sql.catalog.catalog import Catalog
from nl2sql.retrieval.schema_linker import SchemaLinkResult

#: How the model declines. Parsed by the generator; changing it means changing
#: `extract_sql` in generator.py too.
CANNOT_ANSWER_MARKER = "-- CANNOT_ANSWER:"

#: The stub client finds the question by looking for this. Keep them in sync.
QUESTION_MARKER = "Question:"


SYSTEM_PROMPT = """\
You are an expert data analyst who writes SQLite queries against a bank's \
data warehouse. You translate a question into exactly one SQL query.

RULES -- these are enforced by a validator, not suggestions:

1. Output ONE SELECT statement. Never INSERT, UPDATE, DELETE, DROP, ALTER, \
CREATE, ATTACH, or PRAGMA. A query that modifies anything is rejected.
2. Use ONLY the tables and columns given in the schema below. Do not invent \
column names, and do not assume a column exists because it would be \
convenient. Inventing identifiers is the single most common way to fail here.
3. Write SQLite syntax. Dates are TEXT in 'YYYY-MM-DD' form -- compare them \
as strings, or use strftime('%Y', col) to extract a year. There is no EXTRACT, \
DATEPART, or TO_DATE.
4. Return only the columns needed to answer the question. If the question asks \
"how many", return a single COUNT -- not a list of rows for the user to count.
5. Give computed columns a readable alias: COUNT(*) AS client_count, not COUNT(*).
6. If the question cannot be answered from this schema, reply with exactly:
   {cannot_answer} followed by a one-sentence reason. Do not guess a schema \
that would make it answerable.

OUTPUT FORMAT -- a single fenced SQL block and nothing else. No explanation, \
no preamble, no commentary:

```sql
SELECT ...
```
"""


def build_system_prompt() -> str:
    """The role and rules. Identical on every request, so it caches well."""
    return SYSTEM_PROMPT.format(
        cannot_answer=CANNOT_ANSWER_MARKER,
    )


def build_user_prompt(
    question: str,
    catalog: Catalog,
    link: SchemaLinkResult,
    *,
    examples: list[tuple[str, str]] | None = None,
) -> str:
    """Assemble the request."""
    sections: list[str] = []

    schema_text = catalog.render_schema(
        table_names=link.tables or None,
        column_names=link.columns or None,
    )
    sections.append(f"### Database schema\n\n{schema_text}")

    # Value hints: "the question says 'Benesov', which is a value in
    # district.district_name". This removes the guesswork about which column a
    # literal belongs in, and about its exact spelling and capitalisation.
    hints = link.render_value_hints()
    if hints:
        sections.append(
            "### Values found in the question\n\n"
            "These phrases from the question match real values in the database. "
            "Use them exactly as written here.\n\n"
            f"{hints}"
        )

    if examples:
        rendered = "\n\n".join(
            f"Q: {example_question}\n```sql\n{example_sql.strip()}\n```"
            for example_question, example_sql in examples
        )
        sections.append(
            "### Worked examples\n\n"
            "Similar questions and their correct queries. Follow this style.\n\n"
            f"{rendered}"
        )

    # The marker below is what StubClient parses to recover the question.
    sections.append(f"### {QUESTION_MARKER}\n\n{question}")

    return "\n\n".join(sections)


def build_repair_prompt(
    question: str,
    catalog: Catalog,
    link: SchemaLinkResult,
    failed_sql: str,
    error_feedback: str,
    *,
    attempt: int = 1,
) -> str:
    """Ask the model to fix a query that failed."""
    schema_text = catalog.render_schema(
        table_names=link.tables or None,
        column_names=link.columns or None,
    )

    return f"""\
### Database schema

{schema_text}

### Your previous attempt

```sql
{failed_sql.strip()}
```

### Why it failed

{error_feedback}

### Instructions

This is repair attempt {attempt}. Fix the problem above and output the corrected \
query. Check every table and column name against the schema before answering -- \
if a name is not in the schema above, it does not exist. Change only what is \
necessary to fix the error.

Output a single fenced SQL block and nothing else.

### {QUESTION_MARKER}

{question}
"""


ANSWER_SYSTEM_PROMPT = """\
You turn SQL query results into a short, direct answer for someone who asked a \
question in plain English.

RULES:

1. Answer in one or two sentences. No preamble, no restating the question.
2. Use only what is in the results. If they do not answer the question, say so.
3. Format numbers readably: 1,247 not 1247. Money is Czech koruna -- write \
"43,809 CZK".
4. Do not describe the SQL or mention tables, columns, or queries. The user \
sees the query separately; they want the answer.
5. If the result is a single number, lead with it.
6. If many rows were returned, summarise the pattern and mention the count. \
Do not list them all -- the user can see the table.
"""


def build_answer_prompt(
    question: str,
    sql: str,
    columns: list[str],
    rows: list[tuple],
    *,
    truncated: bool = False,
    max_rows_shown: int = 20,
) -> str:
    """Ask the model to phrase the result as an answer."""
    if not rows:
        result_text = "The query ran successfully but matched no rows."
    else:
        header = " | ".join(columns)
        divider = "-" * len(header)
        body = "\n".join(
            " | ".join("NULL" if value is None else str(value) for value in row)
            for row in rows[:max_rows_shown]
        )
        result_text = f"{header}\n{divider}\n{body}"
        if len(rows) > max_rows_shown:
            result_text += f"\n... and {len(rows) - max_rows_shown:,} more rows"

    notes = ""
    if truncated:
        notes = (
            "\n\nNote: the result was capped at the row limit, so this is a "
            "partial result. Say so in your answer."
        )

    return f"""\
### Question

{question}

### Query that was run

```sql
{sql.strip()}
```

### Results ({len(rows):,} row{"s" if len(rows) != 1 else ""})

{result_text}{notes}

### Your answer
"""
