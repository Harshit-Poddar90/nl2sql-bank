"""Turning a result set into a sentence."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from nl2sql.config import Settings, get_settings
from nl2sql.execution.executor import QueryResult
from nl2sql.generation.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from nl2sql.llm.base import LLMClient, LLMResponse
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)

#: Column names that indicate a money value, so the answer can say "CZK".
_MONEY_HINTS = re.compile(
    r"(balance|amount|salary|payment|credited|withdrawn|total|sum|money|czk)",
    re.IGNORECASE,
)
#: Column names that indicate a plain count.
_COUNT_HINTS = re.compile(r"(count|number|n_|total_\w*count|how_many)", re.IGNORECASE)


@dataclass
class AnswerResult:
    """The natural-language answer, and how it was produced."""

    text: str
    #: "template" or "llm". Reported in the API trace and the eval report.
    method: str = "template"
    llm_response: LLMResponse | None = None
    latency_ms: float = 0.0


def format_number(value: object, *, is_money: bool = False) -> str:
    """Render a number the way a person would write it."""
    if isinstance(value, bool) or value is None:
        return str(value)

    if isinstance(value, int):
        rendered = f"{value:,}"
    elif isinstance(value, float):
        # Whole-valued floats are counts that survived an AVG or a ROUND;
        # printing '1,627.0' looks like a bug even though it is not.
        rendered = f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"
    else:
        return str(value)

    return f"{rendered} CZK" if is_money else rendered


class AnswerSynthesizer:
    """Produces the final natural-language answer."""

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm

    def synthesize(
        self,
        question: str,
        sql: str,
        result: QueryResult,
        *,
        force_llm: bool = False,
    ) -> AnswerResult:
        """Describe ``result`` as an answer to ``question``."""
        started = time.perf_counter()

        # -- empty results. Never worth an LLM call, and a model asked to
        # explain zero rows will speculate about why, which is exactly the
        # ungrounded behaviour we are trying to avoid.
        if result.is_empty:
            return AnswerResult(
                text="No rows matched that question. The data may not contain "
                     "anything meeting those criteria.",
                method="template",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if not force_llm:
            templated = self._try_template(question, result)
            if templated is not None:
                return AnswerResult(
                    text=templated,
                    method="template",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

        if self.llm is not None and self.llm.supports_prose:
            try:
                response = self.llm.complete(
                    system=ANSWER_SYSTEM_PROMPT,
                    user=build_answer_prompt(
                        question, sql, result.columns, result.rows,
                        truncated=result.truncated,
                    ),
                    # A touch of temperature: this is prose, and zero produces
                    # oddly clipped phrasing.
                    temperature=0.2,
                    max_tokens=300,
                )
                return AnswerResult(
                    text=response.text.strip(),
                    method="llm",
                    llm_response=response,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except Exception as exc:
                # An answer-phrasing failure must never lose the user their
                # results. Fall through to the generic description.
                log.warning("answer_synthesis_failed", extra={"error": str(exc)})

        return AnswerResult(
            text=self._describe_generically(result),
            method="template",
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- templates ----------------------------------------------------------
    def _try_template(self, question: str, result: QueryResult) -> str | None:
        """Render an answer directly, or return None if the shape needs a model."""

        # Shape 1: one cell. "How many...", "what is the average...".
        if result.is_scalar:
            column = result.columns[0]
            value = result.scalar_value
            is_money = bool(_MONEY_HINTS.search(column)) and not _COUNT_HINTS.search(column)
            rendered = format_number(value, is_money=is_money)

            subject = self._describe_subject(question, column)
            # "account_count" -> "1,627 accounts", not "1,627 account".
            if subject and _COUNT_HINTS.search(column) and value != 1 \
                    and not subject.endswith(("s", "people")):
                subject += "s"
            if subject:
                return f"{rendered} {subject}."
            return f"{rendered}."

        # Shape 2: one row, few columns. A small set of related facts.
        if result.row_count == 1 and len(result.columns) <= 4:
            parts = [
                f"{self._humanise(column)}: "
                f"{format_number(value, is_money=bool(_MONEY_HINTS.search(column)))}"
                for column, value in zip(result.columns, result.rows[0], strict=False)
            ]
            return "; ".join(parts) + "."

        # Shape 3: a two-column ranking -- label plus number. Very common
        # ("which region has the most X"), and the top entry is the answer.
        if len(result.columns) == 2 and result.row_count >= 2:
            label_column, value_column = result.columns
            first_label, first_value = result.rows[0]
            if isinstance(first_value, (int, float)) and not isinstance(first_value, bool):
                is_money = bool(_MONEY_HINTS.search(value_column))
                rendered = format_number(first_value, is_money=is_money)
                return (
                    f"{first_label} leads with {rendered} "
                    f"({self._humanise(value_column)}), out of {result.row_count} "
                    f"{self._humanise(label_column).lower()} values"
                    + (" (results were capped)." if result.truncated else ".")
                )

        # Anything else needs judgement about what to emphasise. Ask the model.
        return None

    @staticmethod
    def _describe_subject(question: str, column: str) -> str:
        """Work out what the number counts, for the sentence."""
        name = column.lower()

        for affix in ("count", "total", "average", "avg", "sum", "max", "min"):
            name = name.removeprefix(affix + "_").removesuffix("_" + affix)

        name = name.strip("_").replace("_", " ").strip()

        if not name or name in {"count", "n", "total", "value", "result"}:
            # Nothing usable in the alias -- take the plural noun the question
            # used, which is nearly always the subject.
            match = re.search(
                r"\b(accounts?|clients?|customers?|loans?|cards?|transactions?|"
                r"districts?|regions?|people|orders?)\b",
                question,
                re.IGNORECASE,
            )
            if match:
                noun = match.group(1).lower()
                return noun if noun.endswith("s") or noun == "people" else noun + "s"
            return ""

        return name

    @staticmethod
    def _humanise(column: str) -> str:
        """'default_rate_pct' -> 'Default rate pct'."""
        return column.replace("_", " ").strip().capitalize()

    @staticmethod
    def _describe_generically(result: QueryResult) -> str:
        """Last-resort description. Always true, never interesting."""
        suffix = " (capped at the row limit)" if result.truncated else ""
        return (
            f"The query returned {result.row_count:,} row"
            f"{'s' if result.row_count != 1 else ''} "
            f"with columns: {', '.join(result.columns)}{suffix}. "
            f"See the table below for the full result."
        )
