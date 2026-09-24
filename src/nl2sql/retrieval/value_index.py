"""Value-based schema linking: matching a question's literals to the data."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from nl2sql.catalog.catalog import Catalog

#: Longest phrase we try to match. "penalty interest for negative balance" is
#: six words, but nobody types that; four covers every place name and product
#: name in this schema.
MAX_NGRAM_WORDS = 4

#: Values this short are skipped unless the column is a code column. Without
#: this, the value 'in' or 'a' matches half of every question.
MIN_VALUE_LENGTH = 2

#: Single words that are real values in this database but also ordinary
#: English, and so produce far more false matches than real ones.
#:
#: The instructive one is "most". There genuinely is a Czech district called
#: Most, so "which region has the most defaulted loans?" was linking to
#: ``district.district_name`` and dragging the district table to the top of the
#: ranking for every superlative question ever asked. The place name is real;
#: it is just overwhelmingly less likely than the English word.
#:
#: Skipping these costs nothing measurable: BM25 and the embedding ranker still
#: surface ``district`` for a question that is genuinely about Most, they just
#: do not get a 2.0-weighted shove towards it.
AMBIGUOUS_VALUES: frozenset[str] = frozenset(
    {
        # real values here, but common words first and foremost
        "credit", "type", "status", "amount", "date", "account", "in", "out",
        "bank", "cash", "interest", "payment", "monthly", "weekly",
        # Czech place names that collide with English words
        "most", "cheb", "louny", "beroun",
        # question words that should never drive schema selection
        "top", "least", "highest", "lowest", "best", "worst", "more", "less",
        "many", "much", "average", "total", "number", "count",
    }
)

#: Columns whose values must never enter the value index.
#:
#: Dates are the problem case. ``_normalise`` turns '1996-12-31' into the words
#: '1996 12 31', so the year in "how many gold cards were issued in 1996?" was
#: matching partway into a specific transaction date. That is a coincidence of
#: string formatting, not evidence about which column the question means, and
#: it was actively misleading the ranker.
_DATE_COLUMN_PATTERN = re.compile(r"(_date|_at|^date$|birth_number)", re.IGNORECASE)
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalise(text: str) -> str:
    """Lower-case and collapse everything that is not a letter or digit."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


@dataclass(frozen=True)
class ValueHit:
    """A phrase from the question that turned out to be a value in the database."""

    phrase: str            # what the user actually typed, e.g. "north moravia"
    value: str             # the stored value it matched, e.g. "north Moravia"
    table: str
    column: str
    exact: bool            # True for a whole-value match, False for a partial one

    @property
    def qualified_column(self) -> str:
        return f"{self.table}.{self.column}"

    def render_hint(self) -> str:
        """The line injected into the prompt."""
        relation = "is the value" if self.exact else "appears in a value"
        return (
            f"-- The question mentions '{self.phrase}', which {relation} "
            f"'{self.value}' in {self.qualified_column}"
        )


class ValueIndex:
    """Lookup from normalised value text to the columns containing it."""

    def __init__(self, entries: list[tuple[str, str, str]]) -> None:
        # Exact lookup: normalised value -> [(original value, table, column)]
        self._exact: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        # Word lookup, for partial matches: a single word from a multi-word
        # value -> the entries containing it. This is how "Moravia" alone finds
        # both "north Moravia" and "south Moravia".
        self._by_word: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

        for value, table, column in entries:
            if _DATE_COLUMN_PATTERN.search(column) or _ISO_DATE_PATTERN.match(value):
                continue

            normalised = _normalise(value)
            if not normalised:
                continue
            if len(normalised) < MIN_VALUE_LENGTH and not column.endswith("_code"):
                continue

            self._exact[normalised].append((value, table, column))

            words = normalised.split()
            if len(words) > 1:
                for word in words:
                    # Only distinctive words earn a partial-match entry. Short
                    # ones and common ones match everything and mean nothing.
                    if len(word) > 3 and word not in AMBIGUOUS_VALUES:
                        self._by_word[word].append((value, table, column))

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> ValueIndex:
        return cls(catalog.value_index_entries())

    @property
    def size(self) -> int:
        return sum(len(v) for v in self._exact.values())

    def search(self, question: str, *, limit: int = 8) -> list[ValueHit]:
        """Find database values mentioned in ``question``."""
        normalised = _normalise(question)
        if not normalised:
            return []

        words = normalised.split()
        hits: list[ValueHit] = []
        consumed: set[int] = set()      # word positions already claimed by a hit
        # One hit per (phrase, table, column). Without this, "bank" matching
        # both 'collection from another bank' and 'remittance to another bank'
        # produced two identical-looking hits pointing at the same column,
        # which then got counted twice by the fusion step and over-weighted it.
        seen: set[tuple[str, str, str]] = set()

        def record(phrase: str, value: str, table: str, column: str, *, exact: bool) -> bool:
            key = (phrase, table, column)
            if key in seen:
                return False
            seen.add(key)
            hits.append(ValueHit(phrase=phrase, value=value, table=table,
                                 column=column, exact=exact))
            return True

        # Longest n-grams first: specific beats general, so "north Moravia"
        # is claimed as a phrase before "north" and "Moravia" are tried alone.
        for size in range(min(MAX_NGRAM_WORDS, len(words)), 0, -1):
            for start in range(len(words) - size + 1):
                positions = set(range(start, start + size))
                if positions & consumed:
                    continue

                phrase = " ".join(words[start : start + size])
                if len(phrase) < MIN_VALUE_LENGTH:
                    continue
                # Single common words are skipped; multi-word phrases are kept
                # even if they contain one, since "cash withdrawal" is specific
                # in a way that "cash" is not.
                if size == 1 and phrase in AMBIGUOUS_VALUES:
                    continue

                matches = self._exact.get(phrase)
                if matches:
                    for value, table, column in matches[:3]:
                        record(phrase, value, table, column, exact=True)
                    consumed |= positions
                    continue

                # No exact hit: is this word part of a multi-word value?
                if size == 1 and len(phrase) > 3:
                    partial = self._by_word.get(phrase)
                    if partial:
                        for value, table, column in partial[:2]:
                            record(phrase, value, table, column, exact=False)
                        consumed |= positions

        # Exact matches first, then longer phrases: both are stronger evidence.
        hits.sort(key=lambda h: (not h.exact, -len(h.phrase)))
        return hits[:limit]

