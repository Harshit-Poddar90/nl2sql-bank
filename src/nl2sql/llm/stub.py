"""A deterministic, rule-based SQL generator. No network, no API key, no model."""

from __future__ import annotations

import re
from dataclasses import dataclass

from nl2sql.config import Settings
from nl2sql.generation.prompts import QUESTION_MARKER
from nl2sql.llm.base import LLMClient, LLMResponse


@dataclass(frozen=True)
class Entity:
    """A thing the question might be counting, with how to reach it."""

    table: str
    #: What to count. COUNT(DISTINCT x) where rows could duplicate across joins.
    count_expr: str
    #: A sensible default projection when the question wants rows, not a number.
    select_columns: str
    #: Extra FROM/JOIN text appended after the table name.
    joins: str = ""


# Entity keywords, checked in order. Longest and most specific first, so
# "credit card" is not swallowed by "credit".
ENTITIES: list[tuple[re.Pattern[str], Entity]] = [
    (
        re.compile(r"\b(standing order|permanent order|recurring payment)s?\b"),
        Entity("permanent_order", "COUNT(*)", "order_id, account_id, amount, purpose"),
    ),
    (
        re.compile(r"\b(transaction|payment|withdrawal|deposit)s?\b"),
        Entity("bank_transaction", "COUNT(*)",
               "transaction_id, account_id, transaction_date, amount, direction"),
    ),
    (
        re.compile(r"\b(credit card|debit card|card)s?\b"),
        Entity("card", "COUNT(*)", "card_id, card_type, issued_date"),
    ),
    (
        re.compile(r"\b(loan|borrowing|mortgage)s?\b"),
        Entity("loan", "COUNT(*)", "loan_id, account_id, amount, duration_months, status"),
    ),
    (
        re.compile(r"\b(district|region|area)s?\b"),
        Entity("district", "COUNT(*)", "district_id, district_name, region, average_salary"),
    ),
    (
        # Balance questions are about accounts, but must be answered from the
        # summary table -- the whole reason that table exists.
        re.compile(r"\b(balance|money|funds|savings|richest|wealthiest)\b"),
        Entity("account_summary", "COUNT(*)",
               "account_id, owner_client_id, current_balance"),
    ),
    (
        re.compile(r"\b(account)s?\b"),
        Entity("account_summary", "COUNT(*)",
               "account_id, owner_client_id, current_balance"),
    ),
    (
        re.compile(r"\b(client|customer|people|person|holder|men|women|male|female)s?\b"),
        Entity("v_client_overview", "COUNT(DISTINCT client_id)",
               "client_id, gender, age_at_1999, district_name"),
    ),
]

#: Filters. Each maps a phrase in the question to a WHERE fragment, names the
#: tables it applies to, and belongs to a mutually-exclusive *group*.
#:
#: The group is the important part. Rules within a group are tried in order and
#: **only the first match is used**. Without that, "how many loans were never
#: repaid?" matches the `defaulted` rule on "never repaid" and *also* the
#: `repaid` rule on "repaid", producing
#: ``WHERE is_defaulted = 1 AND is_defaulted = 0`` -- a query that is valid,
#: runs happily, and always returns zero.
#:
#: That failure is worth dwelling on: it is silent. Nothing errors, nothing is
#: rejected, the user just gets a confidently wrong answer. Ordering within a
#: group also matters -- the more specific pattern ("never repaid") has to come
#: before the more general one ("repaid") that it contains.
FILTERS: list[tuple[re.Pattern[str], str, tuple[str, ...], str]] = [
    # -- group: loan outcome
    (re.compile(r"\b(defaulted|default|unpaid|not repaid|never repaid|bad loan)"),
     "is_defaulted = 1", ("loan", "v_loan_overview"), "loan_outcome"),
    (re.compile(r"\b(repaid|paid in full|good loan|successful)"),
     "is_defaulted = 0", ("loan", "v_loan_overview"), "loan_outcome"),
    # -- group: loan lifecycle
    (re.compile(r"\b(still running|active|ongoing|not finished)"),
     "is_finished = 0", ("loan", "v_loan_overview"), "loan_lifecycle"),
    (re.compile(r"\b(finished|completed|closed)"),
     "is_finished = 1", ("loan", "v_loan_overview"), "loan_lifecycle"),
    # -- group: gender
    (re.compile(r"\b(female|women|woman)\b"),
     "gender = 'female'", ("client", "v_client_overview"), "gender"),
    (re.compile(r"\b(male|men|man)\b"),
     "gender = 'male'", ("client", "v_client_overview"), "gender"),
    # -- group: card tier
    (re.compile(r"\bgold\b"), "card_type = 'gold'", ("card",), "card_type"),
    (re.compile(r"\bclassic\b"), "card_type = 'classic'", ("card",), "card_type"),
    (re.compile(r"\bjunior\b"), "card_type = 'junior'", ("card",), "card_type"),
    # -- group: money direction
    (re.compile(r"\b(withdrawal|withdrew|withdrawn|taken out|outgoing|spent)"),
     "direction = 'withdrawal'", ("bank_transaction",), "direction"),
    (re.compile(r"\b(credited|paid in|incoming|deposit)"),
     "direction = 'credit'", ("bank_transaction",), "direction"),
]

#: The numeric column each entity means when a threshold is mentioned.
THRESHOLD_COLUMN: dict[str, str] = {
    "account_summary": "current_balance",
    "loan": "amount",
    "v_loan_overview": "amount",
    "bank_transaction": "amount",
    "permanent_order": "amount",
    "district": "average_salary",
    "v_client_overview": "owned_account_balance",
}

#: Columns to group by, when the question asks for a breakdown.
GROUP_BY: list[tuple[re.Pattern[str], str, tuple[str, ...]]] = [
    (re.compile(r"\bby region\b|\bper region\b|\beach region\b|\bwhich region\b"),
     "region", ("district", "v_client_overview", "v_loan_overview", "v_account_overview")),
    (re.compile(r"\bby district\b|\bper district\b|\beach district\b"),
     "district_name",
     ("district", "v_client_overview", "v_loan_overview", "v_account_overview")),
    (re.compile(r"\bby gender\b|\bper gender\b"), "gender",
     ("client", "v_client_overview")),
    (re.compile(r"\bby status\b|\bper status\b|\beach status\b"), "status",
     ("loan", "v_loan_overview")),
    (re.compile(r"\bby (card )?type\b"), "card_type", ("card",)),
]

_NUMBER = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\b")
_MORE_THAN = re.compile(r"\b(more than|greater than|over|above|at least|exceed(?:ing|s)?)\b")
_LESS_THAN = re.compile(r"\b(less than|under|below|fewer than|at most)\b")
_TOP_N = re.compile(r"\b(?:top|first|highest|largest|richest|biggest)\s+(\d+)\b")


class StubClient(LLMClient):
    """Rule-based SQL generation. Deterministic, offline, free."""

    provider = "stub"

    #: This generator emits SQL and nothing else. The answer synthesizer must
    #: fall back to its own templates rather than asking this for a sentence.
    supports_prose = False

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model = "rule-based-stub-v1"

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        question = self._extract_question(user)
        sql = self.generate_sql(question)

        return LLMResponse(
            text=f"```sql\n{sql}\n```",
            provider=self.provider,
            model=self.model,
            # Token counts are 0 rather than estimated. Reporting a made-up
            # number would put fictional costs into the benchmark table.
            input_tokens=0,
            output_tokens=0,
        )

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Pull the user's question back out of the assembled prompt."""
        if QUESTION_MARKER in prompt:
            tail = prompt.rsplit(QUESTION_MARKER, 1)[1]
            # The question is the first non-empty line after the marker.
            for line in tail.splitlines():
                if line.strip():
                    return line.strip()
        return prompt.strip().splitlines()[-1] if prompt.strip() else ""

    # -- the four decisions -------------------------------------------------
    @staticmethod
    def _detect_entity(question: str) -> Entity:
        for pattern, entity in ENTITIES:
            if pattern.search(question):
                return entity
        # Nothing recognised: accounts are the most common subject here.
        return ENTITIES[-2][1]

    @staticmethod
    def _detect_filters(question: str, entity: Entity) -> list[str]:
        clauses: list[str] = []
        # First match wins within each group, so contradictory filters can
        # never both be emitted.
        used_groups: set[str] = set()
        for pattern, clause, applicable_tables, group in FILTERS:
            if group in used_groups:
                continue
            if entity.table in applicable_tables and pattern.search(question):
                clauses.append(clause)
                used_groups.add(group)

        # A numeric threshold: "more than 50000", "under 1000".
        column = THRESHOLD_COLUMN.get(entity.table)
        if column:
            numbers = [
                float(match.replace(",", "")) for match in _NUMBER.findall(question)
            ]
            # Ignore bare years -- "in 1996" is a date filter, not a threshold.
            numbers = [n for n in numbers if not (1900 <= n <= 2100 and n == int(n))]
            if numbers:
                threshold = numbers[0]
                rendered = f"{threshold:g}"
                if _MORE_THAN.search(question):
                    clauses.append(f"{column} > {rendered}")
                elif _LESS_THAN.search(question):
                    clauses.append(f"{column} < {rendered}")

        # A year: "in 1996".
        year_match = re.search(r"\b(19\d{2})\b", question)
        if year_match:
            date_column = {
                "loan": "loan_date",
                "v_loan_overview": "loan_date",
                "bank_transaction": "transaction_date",
                "card": "issued_date",
                "account": "opened_date",
            }.get(entity.table)
            if date_column:
                clauses.append(
                    f"strftime('%Y', {date_column}) = '{year_match.group(1)}'"
                )

        return clauses

    @staticmethod
    def _detect_group_by(question: str, entity: Entity) -> str | None:
        for pattern, column, applicable_tables in GROUP_BY:
            if entity.table in applicable_tables and pattern.search(question):
                return column
        return None

    def generate_sql(self, question: str) -> str:
        """Compose a query from the four decisions. The whole generator."""
        lowered = question.lower()
        entity = self._detect_entity(lowered)
        filters = self._detect_filters(lowered, entity)
        group_by = self._detect_group_by(lowered, entity)

        wants_count = bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", lowered))
        wants_average = bool(re.search(r"\baverage\b|\bmean\b|\btypical\b", lowered))
        wants_total = bool(re.search(r"\btotal\b|\bsum\b|\bcombined\b", lowered))
        wants_max = bool(re.search(r"\b(highest|largest|maximum|richest|most money)\b", lowered))
        wants_min = bool(re.search(r"\b(lowest|smallest|minimum|poorest)\b", lowered))

        numeric_column = THRESHOLD_COLUMN.get(entity.table, "amount")

        # -- projection
        if wants_count or (group_by and not (wants_average or wants_total)):
            projection = entity.count_expr + " AS count"
        elif wants_average:
            projection = f"ROUND(AVG({numeric_column}), 2) AS average_{numeric_column}"
        elif wants_total:
            projection = f"ROUND(SUM({numeric_column}), 2) AS total_{numeric_column}"
        elif wants_max and not _TOP_N.search(lowered):
            projection = f"MAX({numeric_column}) AS max_{numeric_column}"
        elif wants_min and not _TOP_N.search(lowered):
            projection = f"MIN({numeric_column}) AS min_{numeric_column}"
        else:
            projection = entity.select_columns

        if group_by:
            projection = f"{group_by}, {projection}"

        parts = [f"SELECT {projection}", f"FROM {entity.table}{entity.joins}"]

        if filters:
            parts.append("WHERE " + "\n  AND ".join(filters))

        if group_by:
            parts.append(f"GROUP BY {group_by}")
            parts.append("ORDER BY 2 DESC")

        # -- ordering and limit for "top N" style questions
        top_n = _TOP_N.search(lowered)
        if top_n:
            direction = "ASC" if wants_min else "DESC"
            if not group_by:
                parts.append(f"ORDER BY {numeric_column} {direction}")
            parts.append(f"LIMIT {top_n.group(1)}")
        elif not (wants_count or wants_average or wants_total or wants_max or wants_min):
            # A bare listing question. Cap it -- the guardrail layer would
            # anyway, but emitting a bounded query is better manners.
            parts.append("LIMIT 20")

        return "\n".join(parts) + ";"
