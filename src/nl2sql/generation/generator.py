"""SQL generation: question in, candidate query out."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from nl2sql.catalog.catalog import Catalog
from nl2sql.config import Settings, get_settings
from nl2sql.exceptions import SQLGenerationError
from nl2sql.generation.few_shot import FewShotSelector
from nl2sql.generation.prompts import (
    CANNOT_ANSWER_MARKER,
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
)
from nl2sql.llm.base import LLMClient, LLMResponse
from nl2sql.logging_setup import get_logger
from nl2sql.retrieval.schema_linker import SchemaLinker, SchemaLinkResult

log = get_logger(__name__)

#: A fenced block. Non-greedy so each fence is captured separately when a reply
#: contains several.
#:
#: The language tag is deliberately *not* enumerated here. An alternation like
#: ``(?:sql|sqlite|postgres)?`` looks right and is subtly broken: regex
#: alternation is first-match, not longest-match, so ``sql`` matches the start
#: of ```` ```sqlite ```` and leaves ``ite`` glued to the front of the query.
#: The result is a syntax error on a query the model got completely right.
#:
#: Matching any fence and stripping a leading tag line afterwards
#: (:func:`_strip_language_tag`) avoids the whole class of problem and handles
#: tags nobody has thought of yet.
_FENCE_PATTERN = re.compile(r"```(.*?)```", re.DOTALL)

#: A lone word on the first line of a fence is a language tag, not SQL. No SQL
#: statement is a single bare word, so this cannot swallow a real query.
_LANGUAGE_TAG_PATTERN = re.compile(r"^[ \t]*[A-Za-z][A-Za-z0-9_+#-]*[ \t]*\r?\n")

#: Fallback for replies with no fence at all: find where SQL starts and take
#: everything from there.
_BARE_SQL_PATTERN = re.compile(r"\b(WITH|SELECT)\b.*", re.DOTALL | re.IGNORECASE)

#: Conversational lead-ins some models emit despite instructions to the contrary.
_PREAMBLE_PATTERN = re.compile(
    r"^\s*(?:here(?:'s| is)[^\n]*|sure[^\n]*|certainly[^\n]*|"
    r"the (?:following )?query[^\n]*)\n+",
    re.IGNORECASE,
)


@dataclass
class GenerationResult:
    """A generated query, plus how it was produced."""

    sql: str
    #: Set instead of ``sql`` when the model declined. ``sql`` is empty then.
    refusal_reason: str | None = None

    link: SchemaLinkResult = field(default_factory=SchemaLinkResult)
    examples_used: list[str] = field(default_factory=list)
    llm_response: LLMResponse | None = None

    latency_ms: float = 0.0
    #: Which attempt this was: 0 is the first try, 1+ are repairs.
    attempt: int = 0

    @property
    def refused(self) -> bool:
        return self.refusal_reason is not None

    def summary(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "attempt": self.attempt,
            "refused": self.refused,
            "tables": ",".join(self.link.tables),
            "examples": len(self.examples_used),
            "latency_ms": round(self.latency_ms, 1),
        }
        if self.llm_response:
            payload.update(
                {
                    "input_tokens": self.llm_response.input_tokens,
                    "output_tokens": self.llm_response.output_tokens,
                }
            )
        return payload


def extract_sql(text: str) -> tuple[str, str | None]:
    """Pull the SQL out of a model reply."""
    if not text or not text.strip():
        raise SQLGenerationError(
            "The model returned an empty response.",
            user_message="The model did not produce a query. Try rephrasing the question.",
        )

    # Refusal is checked first and against the raw text, because the marker is
    # a SQL comment and may legitimately appear inside a fenced block.
    if CANNOT_ANSWER_MARKER in text:
        _, _, reason = text.partition(CANNOT_ANSWER_MARKER)
        # Strip comment markers off the continuation lines of a multi-line reason.
        cleaned = " ".join(
            line.strip().lstrip("-").strip()
            for line in reason.strip().splitlines()
            if line.strip()
        )
        cleaned = cleaned.split("```")[0].strip()
        return "", cleaned or "The question cannot be answered from this schema."

    candidate: str | None = None

    fenced = [_strip_language_tag(block) for block in _FENCE_PATTERN.findall(text)]
    if fenced:
        # Prefer the first block that actually looks like a query. Models
        # sometimes emit a results table in a second fence, or an explanatory
        # block first.
        for block in fenced:
            if re.search(r"\b(SELECT|WITH)\b", block, re.IGNORECASE):
                candidate = block
                break
        candidate = candidate or fenced[0]
    else:
        stripped = _PREAMBLE_PATTERN.sub("", text)
        match = _BARE_SQL_PATTERN.search(stripped)
        if match:
            candidate = match.group(0)

    if candidate is None:
        raise SQLGenerationError(
            f"No SQL found in the model response: {text[:300]!r}",
            user_message="The model's reply did not contain a SQL query.",
        )

    return normalise_sql(candidate), None


def _strip_language_tag(block: str) -> str:
    """Remove a leading ```` ```sql ```` style language tag from a fenced block."""
    if _LANGUAGE_TAG_PATTERN.match(block):
        return _LANGUAGE_TAG_PATTERN.sub("", block, count=1)
    return block


def normalise_sql(sql: str) -> str:
    """Tidy a query into a canonical form."""
    cleaned = sql.strip()

    # Drop any prose that followed the query outside a fence.
    cleaned = re.sub(
        r"\n\s*(?:This query|The query|Note:|Explanation:).*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    cleaned = cleaned.strip().rstrip(";").strip()

    # Collapse trailing whitespace per line, keep the indentation the model
    # chose -- readable SQL is part of what the user is shown.
    return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()


class SQLGenerator:
    """Turns questions into candidate SQL."""

    def __init__(
        self,
        catalog: Catalog,
        linker: SchemaLinker,
        llm: LLMClient,
        few_shot: FewShotSelector | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog
        self.linker = linker
        self.llm = llm
        self.few_shot = few_shot or FewShotSelector.from_settings(self.settings)
        self._system_prompt = build_system_prompt()

    def generate(self, question: str) -> GenerationResult:
        """First attempt at a question."""
        started = time.perf_counter()

        link = self.linker.link(question)
        examples = self.few_shot.select(question)

        user_prompt = build_user_prompt(
            question,
            self.catalog,
            link,
            examples=[e.as_pair() for e in examples],
        )

        response = self.llm.complete(system=self._system_prompt, user=user_prompt)
        sql, refusal = extract_sql(response.text)

        result = GenerationResult(
            sql=sql,
            refusal_reason=refusal,
            link=link,
            examples_used=[e.question for e in examples],
            llm_response=response,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempt=0,
        )
        log.info("sql_generated", extra=result.summary())
        return result

    def repair(
        self,
        question: str,
        failed_sql: str,
        error_feedback: str,
        previous: GenerationResult,
        *,
        attempt: int = 1,
    ) -> GenerationResult:
        """Show the model its error and ask for a corrected query."""
        started = time.perf_counter()

        prompt = build_repair_prompt(
            question,
            self.catalog,
            previous.link,
            failed_sql,
            error_feedback,
            attempt=attempt,
        )

        response = self.llm.complete(system=self._system_prompt, user=prompt)
        sql, refusal = extract_sql(response.text)

        result = GenerationResult(
            sql=sql,
            refusal_reason=refusal,
            link=previous.link,
            examples_used=previous.examples_used,
            llm_response=response,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempt=attempt,
        )
        log.info("sql_repaired", extra=result.summary())
        return result
