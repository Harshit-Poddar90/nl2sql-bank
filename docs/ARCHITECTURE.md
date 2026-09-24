# Architecture

This document explains *why* the system is built the way it is. The README covers what it does;
this covers the reasoning, the trade-offs, and the things that turned out to be harder than they
looked.

Read [`src/nl2sql/pipeline.py`](../src/nl2sql/pipeline.py) alongside this — it is the orchestrator,
and every stage below is a call in it.

---

## 1. The shape of the problem

Text-to-SQL looks like a single prompt-engineering task and is not. It decomposes into four
problems with different failure modes:

| Problem | Failure mode | Handled in |
|---|---|---|
| Which part of the schema is relevant? | model writes about the wrong tables | `retrieval/` |
| What does this schema *mean*? | model misreads `disposition_type` | `catalog/semantics.yaml` |
| Is the generated SQL safe to run? | `DROP TABLE`, runaway joins | `guardrails/` |
| Was the answer actually right? | you have no idea | `evaluation/` |

The LLM call sits in the middle and is the least interesting part. It is roughly 40 lines
(`generation/generator.py`). Everything else is the system.

---

## 2. Data layer

### The source data is hostile

The Berka dataset is real, which means it is messy in ways synthetic data never is:

| Problem | Example | Fix |
|---|---|---|
| Dates are integers | `930101` | → `'1993-01-01'` ISO text |
| Enums are Czech | `POPLATEK MESICNE` | → paired English label |
| Columns named `A1`..`A16` | `A11` | → `average_salary` |
| Gender hidden in an ID | `706213` | → `('1970-12-13', 'female')` |
| Reserved words as tables | `order`, `trans` | → `permanent_order`, `bank_transaction` |
| Three spellings of NULL | `''`, `' '`, `'?'` | → `NULL` |

**Why both the code and the label.** Every coded column is stored twice: `status_code = 'B'` *and*
`status = 'finished, not paid (defaulted)'`. It costs storage. It buys two things: the model can
filter on whichever reads more naturally to it, and results stay comparable with published work on
this dataset, which uses the original codes.

**Why gender decoding matters.** Czech national IDs add 50 to the birth month for women. Without
decoding it, *"how many female clients took out a loan?"* is unanswerable — not hard, literally
impossible. One rule in the ETL turns an entire class of question from impossible to trivial. That
is the highest-leverage work in the project, and it happens before any model is involved.

### The materialised `account_summary`

The single most natural question anyone asks this database is *"how many accounts have more than
X?"*. Answering it from `bank_transaction` requires finding each account's latest balance:

```sql
SELECT COUNT(*) FROM (
  SELECT account_id, balance_after,
         ROW_NUMBER() OVER (PARTITION BY account_id
                            ORDER BY transaction_date DESC, transaction_id DESC) rn
  FROM bank_transaction
) WHERE rn = 1 AND balance_after > 50000
```

A window function over 1,056,320 rows. **Measured: 989 ms.**

The same question against the precomputed table:

```sql
SELECT COUNT(*) FROM account_summary WHERE current_balance > 50000
```

**Measured: 1.1 ms.** A 900× speedup, and it uses a covering index.

Both return 1,627.

The trade-off is staleness — `account_summary` is correct as of the last ETL run. For a static
historical dataset that cost is exactly zero. For a live warehouse you would rebuild it
incrementally, and the honest thing is to say so rather than pretend materialisation is free.

The semantic catalog tells the model this table is the fast path, and the few-shot examples
demonstrate it. Both are necessary: telling it once in a rule is weaker than showing it three times
in worked examples.

### Verification

Sixteen checks run after every build:

- **Row counts** against the dataset's published figures
- **Referential integrity** via `PRAGMA foreign_key_check`
- **Semantic invariants** — every account has exactly one `OWNER`; every `account_summary` balance
  equals the account's last transaction balance; `is_defaulted` agrees with `status_code`
- **Date ranges** — transactions must start `1993-01-01` and end `1998-12-31`, which catches a
  broken date parser immediately

The loan outcome split comes out as **606 repaid / 76 defaulted**, matching the dataset's
documented figures exactly. That is the strongest single signal that the ETL is correct: it is an
external ground truth the pipeline could not have accidentally reproduced.

---

## 3. Semantic catalog

Introspection tells a model what columns *exist*. It cannot tell it what they *mean*.

```yaml
disposition:
  notes:
    - "THIS IS THE MOST COMMON SOURCE OF WRONG ANSWERS. Every account has exactly one
       OWNER and may also have one DISPONENT. Counting rows here counts *relationships*,
       not people and not accounts."
```

Nothing in a `CREATE TABLE` statement conveys that. Without it, *"how many customers have an
account?"* returns 5,369 (every disposition row) instead of 4,500 (actual owners) — a plausible
number, confidently wrong, with nothing anywhere to flag it.

`semantics.yaml` carries four kinds of knowledge:

1. **Descriptions** — what a table or column is
2. **Synonyms** — "customer", "client", "people", "account holder" all mean `client`
3. **Business rules** — "defaulted means `is_defaulted = 1`, which is status B *or* D, not just B"
4. **Join hints** — written as literal SQL fragments, because a model given
   *"client relates to account via disposition"* still guesses the column names, and one given the
   actual `ON` clause does not

### It is validated against the live schema

If `semantics.yaml` describes a column that no longer exists, `build_catalog()` raises
`CatalogError` and the process stops. This is deliberate. A semantic layer that has silently
drifted from the schema produces subtly worse SQL with no error anywhere — the worst possible
failure mode, because there is nothing to debug.

### Two distinct value limits

Profiling records distinct values, but with **two different thresholds**, and conflating them was
a real bug during development:

| Limit | Value | Used for |
|---|---|---|
| `ENUM_MAX_DISTINCT` | 40 | values rendered *into the prompt* |
| `VALUE_INDEX_MAX_DISTINCT` | 200 | values in the *retrieval index* |

`district_name` has 77 values. Far too many to list in every prompt; absolutely necessary for
"Benesov" to link to that column. One limit forces a choice between a bloated prompt and a
retriever blind to place names. Two limits gets both.

---

## 4. Retrieval

### Why retrieve at all with 12 tables?

Honestly — you don't need to. The full schema fits comfortably in a context window, and the eval
harness measures exactly this as the `--no-retrieval` ablation.

It is here because **it is the part that does not collapse when the schema is not this size**. At
500 tables the full schema does not fit at any price, and this module is what stands between the
system and that wall. Building it against 12 tables, where every answer is checkable by hand, is
the right way to find out whether it works.

### Three retrievers, fused

```
question ──┬──> BM25          ──> ranked list ──┐
           ├──> embeddings    ──> ranked list ──┼──> RRF ──> tables + columns
           └──> value index   ──> ranked list ──┘
```

**Why RRF rather than a weighted sum of scores?** Because a BM25 score of 8.2 and a cosine
similarity of 0.71 are not on comparable scales and cannot be meaningfully added. RRF reads only
each retriever's *ordering*:

```
score(d) = Σ  weight / (K + rank(d))
```

It is also unreasonably robust — one retriever returning nonsense degrades the result gracefully
instead of poisoning it.

### The value index, and two bugs it taught

The value index matches question phrases against actual data values. It is the only way to connect
"Benesov" to `district_name`, or "gold" to `card_type`.

It also produced two instructive false positives, both caught by inspecting real output:

**There is a Czech district called Most.** So *"which region has the **most** defaulted loans?"*
matched `district.district_name` and — with the value index weighted 2.0, the highest of the three —
dragged the `district` table to the top of the ranking for every superlative question ever asked.
The fix is an ambiguity list: common English words are skipped as single-word matches. The place
name is real; it is just overwhelmingly less likely than the English word.

**Years matched into dates.** Normalisation turns `1996-12-31` into the words `1996 12 31`, so
*"how many gold cards were issued in 1996?"* matched partway into a specific transaction date.
That is a coincidence of string formatting, not evidence about column relevance. Date columns are
now excluded from the index entirely.

Neither bug caused an error. Both quietly degraded ranking quality, which is exactly the kind of
thing that never surfaces unless you look.

### Foreign-key closure

After selecting the top-k tables, any table they reference via a foreign key is pulled in. Without
it, retrieval returns `loan` but not `account`, and the model cannot write the join it needs. Cheap
insurance against a structurally impossible query.

---

## 5. Generation

### Prompt structure

Schema, then value hints, then worked examples, then the question — in that order, because the
question is what the model should still be looking at when it starts writing.

### Refusal is a defined output

```
-- CANNOT_ANSWER: The bank stores no names or contact details for clients.
```

Without an explicit escape hatch, a model asked for `client.name` invents that column *every single
time*, because producing something is the strongest pull in its training. Giving refusal a defined
syntax converts a confident wrong answer into an honest one.

The few-shot selector **guarantees a refusal example is always included**, replacing the weakest
pick if necessary. A model that never sees the refusal pattern never uses it.

### Few-shot selection with a diversity penalty

Straight top-k on a bank containing three balance questions returns all three, and the model learns
one pattern three times. Maximal Marginal Relevance applies a similarity penalty against
already-selected examples — trading a little relevance for a lot more coverage.

### Extraction is not trivial

Models are told to return a bare fenced block. Observed in practice: prose preambles, two fences
where the second is an explanation, no fence at all, `sqlite` instead of `sql` as the tag, trailing
"This query returns..." sentences.

Every shape the extractor cannot handle is a **silent cap on accuracy** — a correct query thrown
away because of how it was wrapped.

One bug worth recording: the fence pattern originally enumerated language tags as
`(?:sql|sqlite|postgres)?`. Regex alternation is **first-match, not longest-match**, so `sql`
matched the start of ` ```sqlite ` and left `ite` glued to the front of the query — a syntax error
on a query the model got completely right. The fix is to match any fence and strip a leading
one-word line afterwards, which also handles tags nobody has thought of yet.

---

## 6. Guardrails

### AST, never strings

```python
FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop,
                   exp.Create, exp.Alter, exp.TruncateTable, exp.Merge, exp.Command)
```

A regex blocklist for `DROP` is defeated by `/**/DrOp`, by `D/**/ROP`, by case, by encoding — and
false-positives on a comment containing the word. A parser has none of those blind spots. If
sqlglot says the node is a `Select`, it is a `Select`, whatever it looked like.

`exp.Command` is the catch-all sqlglot uses for statements it has no dedicated node for: `PRAGMA`,
`ATTACH`, `VACUUM`. Rejecting the whole class means new vendor-specific statements are blocked by
default rather than needing to be enumerated.

### Writes are checked everywhere, not just at the root

```sql
WITH gone AS (DELETE FROM loan RETURNING *) SELECT * FROM gone
```

The top-level node is a `Select`. A check that only inspected the statement type waves this
through. `find_all()` walks the entire tree.

### The identifier allowlist

Every table and column must exist in the catalog. This is what catches hallucinated schema — the
most common real failure — *before* the database produces a confusing error:

```
Unknown column(s): 'current_balence' (did you mean 'current_balance'?)
```

The suggestion comes from `difflib` and goes straight into the repair prompt, which is why most
repairs succeed on the first retry.

Names the query defines itself are allowed, but only where they can appear: a CTE name can stand
in for a table, while table, subquery and output aliases can only stand in for columns. An earlier
version treated every alias as a valid table name, so `SELECT * FROM sqlite_master AS
sqlite_master` walked straight past the allowlist. Being stricter than this would reject every
well-written query; being looser is a hole.

### Cross joins: what sqlglot taught me

The obvious rule is *"allow an explicit `CROSS JOIN` since the author meant it, reject an implicit
`FROM a, b`"*. That rule **cannot be implemented**, because sqlglot normalises comma-joins into
exactly the node an explicit `CROSS JOIN` produces — `kind='CROSS'`, no `ON`. By the time you see
the AST they are indistinguishable.

That turns out to be the right answer anyway. The danger of a cross join is the row count, and
`FROM a, b` and `a CROSS JOIN b` produce identical row counts. Intent does not make a
1,056,320 × 5,369 result set smaller. So the rule is about size, not syntax:

- unconditioned join touching a table with ≥100k rows → **rejected**
- unconditioned join across 3+ tables → **rejected**
- two small tables → allowed, with a warning

### `LIMIT` is injected into the AST

Not appended to the string. String appending breaks on a query that already has a `LIMIT`, on one
ending in a comment, and on set operations.

A bug caught here is worth recording, because it is the exact failure mode a security
layer must never have. `UNION` queries cannot take a `LIMIT` directly, so they are wrapped:
`SELECT * FROM (<union>) LIMIT n`. The original code called `statement.replace(wrapped)` — but
`.replace()` on a **root** node does not update the caller's reference. The validator therefore
reported `limit_applied=1000, limit_was_injected=True` while emitting SQL with **no `LIMIT` in it
at all**.

It reported success while doing nothing. The fix is for `_apply_row_limit` to *return* the node to
render, making the substitution impossible to miss.

### Defence in depth

The validator is not trusted alone. The executor opens SQLite with `mode=ro` in the URI, so the
operating system refuses writes:

```
sqlite3.OperationalError: attempt to write a readonly database
```

Two independent mechanisms must fail before anything is damaged.

---

## 7. Execution

**SQLite has no statement timeout.** The usual approach is to hope queries are fast. Instead, a
*progress handler* is registered — a callback SQLite invokes every 10,000 virtual-machine
instructions, which aborts the statement by returning non-zero:

```python
def _abort_if_expired() -> int:
    return 1 if time.monotonic() > deadline else 0

raw.set_progress_handler(_abort_if_expired, 10_000)
```

This interrupts a query *mid-scan*, which is the only way to actually stop one. Verified: a
deliberately expensive self-join on the million-row transaction table is killed at **2.00 s**
against a 2 s budget.

The handler is removed afterwards, so a pooled connection is never left armed.

**Truncation detection** fetches `max_rows + 1`. Knowing whether a 1,000-row result was *all* of
them changes the answer — "1,000 accounts" and "at least 1,000 accounts" are different claims.

---

## 8. The repair loop

Both failure paths feed it, because from the model's point of view a guardrail rejection and a
database error are the same thing: a specific, actionable description of what was wrong.

```
generate → validate → execute
              ↓          ↓
           rejected    failed
              └────┬─────┘
                   ↓
        show the model its own error
                   ↓
              regenerate  (bounded)
```

The database's own message is the most valuable signal available. `no such column: account.balance`
tells the model precisely what to fix, far more effectively than any generic "please try again".

**The schema link is reused, not recomputed.** The question has not changed, so the relevant tables
have not either — re-running retrieval only adds latency and risks drifting to a different table
set mid-repair.

**Bounded at 2 attempts.** An unbounded retry loop against a paid API is how a bad question becomes
a large invoice.

Every attempt — including the failures — is recorded in `AskResult.attempts` and shown in the UI's
trace panel and the audit log.

---

## 9. Evaluation

### Execution accuracy, and why the alternatives are worse

*String comparison* fails immediately: `COUNT(*)` and `COUNT(1)` are the same query written twice.

*AST exact-match* is better but still wrong: a subquery and a join can be semantically identical
and structurally unrelated.

What the user receives is the answer, so the answer is what gets scored: run both queries, compare
result sets.

### Three things that make comparison subtle

Getting any of these wrong produces a plausible-looking score that is several points off —
much harder to notice than an obvious break.

| Problem | Handling |
|---|---|
| **Row order** | `GROUP BY` has no defined order. Enforced only when the gold query's `ORDER BY` has a `LIMIT`, i.e. when the order genuinely determines *which rows* are returned |
| **Column order** | `SELECT a, b` and `SELECT b, a` answer the same question. On a direct-match failure, try every permutation of whole columns. An earlier version sorted each column independently, which broke row pairing and scored `(a,1),(b,2)` equal to `(a,2),(b,1)`: a wrong answer counted as right |
| **Numeric type** | `5369` vs `5369.0`; `43808.94` vs `43808.9400000001`. Normalised to 2 decimal places |

There is also a mundane trap: sorting rows for multiset comparison crashes with `TypeError` when a
column mixes `None` with numbers, which real result sets do constantly. Sorting by
`(type_name, str(value))` gives an arbitrary but *consistent* order, which is all a multiset
comparison needs.

### Metrics, and why each exists

| Metric | Answers |
|---|---|
| **Execution accuracy** | how often is the answer right? |
| Valid SQL rate | how often does it produce *runnable* SQL? |
| Refusal accuracy | does it decline the unanswerable questions? |
| Guardrail block rate | how often does the safety layer fire? |
| Repair rate / repair success rate | does self-correction earn its place? |
| p50 / p95 latency | is it usable? |
| Cost per question | is it affordable at scale? |

**The gap between execution accuracy and valid SQL rate is the most useful number in the report.**
It separates "wrote broken SQL" from "wrote working SQL that answers the wrong question". Those
need completely different fixes: the first is a prompting or schema-rendering problem, the second
is a semantic understanding problem.

### Failure classification

Every failure is bucketed into a mode that implies a different fix:

| Bucket | What to do about it |
|---|---|
| `missed_refusal` | not cautious enough — it invented a schema |
| `refused_answerable` | too cautious — the refusal instruction is too strong |
| `guardrail_blocked` | hallucinated identifiers |
| `execution_error` | valid-looking SQL the database rejected |
| `timeout` | query too expensive |
| `wrong_result` | ran fine, answered a different question — the hard one |

"62% accuracy" is a number. "62% accuracy, and 40% of failures are the model reaching for
`bank_transaction` instead of `account_summary`" is something you can act on.

### The baseline is a real component

The offline generator is a 200-line semantic parser that decomposes a question into entity,
aggregate, filters and shape, then composes SQL. Its measured profile:

| Difficulty | Accuracy |
|---|---:|
| easy | 95.0% |
| medium | 20.0% |
| hard | 0.0% |
| refusal | 0.0% |
| **overall** | **36.9%** |

That is the honest ceiling of pattern matching on this benchmark, and it reframes the headline
number: the question is not "does the LLM get X%" but "how much of the 63% that rules cannot touch
does it recover?"

It also caught a bug worth recording. `"how many loans were never repaid?"` matched both the
*defaulted* pattern (on "never repaid") and the *repaid* pattern (on "repaid"), producing:

```sql
WHERE is_defaulted = 1 AND is_defaulted = 0
```

Valid SQL. Runs happily. Always returns zero. **Nothing errors** — the user just gets a confidently
wrong answer. Filters are now grouped so only the first match in each mutually-exclusive group
applies.

---

## 10. Things I would do differently at larger scale

Stated plainly, because every design has a boundary:

- **`account_summary` would need incremental refresh.** A full rebuild is fine for a static
  dataset; a live warehouse needs CDC or a scheduled incremental merge.
- **The value index would not fit in memory.** At 200 distinct values per column across 500 tables
  it belongs in a proper inverted index, not a Python dict.
- **Embeddings would need a real vector store.** A NumPy matrix and a dot product is correct at 151
  documents and wrong at 100,000.
- **The audit log would go to a warehouse, not SQLite.** It is append-only and grows without bound.
- **Retrieval would need its own evaluation set.** Right now retrieval quality is measured only
  indirectly, through end-to-end accuracy. At scale you want recall@k on a labelled set of
  question → relevant-table pairs, because a retrieval regression and a generation regression look
  identical from the outside.
- **Postgres over SQLite**, with a role that has `SELECT` and nothing else — stronger than anything
  enforceable from the client.

---

## Reading order

If you are coming to this cold:

1. [`pipeline.py`](../src/nl2sql/pipeline.py) — the orchestrator, and the shape of everything
2. [`guardrails/validator.py`](../src/nl2sql/guardrails/validator.py) — the security boundary
3. [`catalog/semantics.yaml`](../src/nl2sql/catalog/semantics.yaml) — the domain knowledge
4. [`retrieval/schema_linker.py`](../src/nl2sql/retrieval/schema_linker.py) — the RAG layer
5. [`evaluation/metrics.py`](../src/nl2sql/evaluation/metrics.py) — how correctness is judged
6. [`data/etl.py`](../src/nl2sql/data/etl.py) — where the data quality comes from
