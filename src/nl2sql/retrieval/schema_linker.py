"""Schema linking: deciding which slice of the database the question is about."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from nl2sql.catalog.catalog import Catalog
from nl2sql.config import Settings, get_settings
from nl2sql.logging_setup import get_logger
from nl2sql.retrieval.bm25 import BM25Index
from nl2sql.retrieval.embeddings import Embedder, EmbeddingCache, get_embedder
from nl2sql.retrieval.value_index import ValueHit, ValueIndex

log = get_logger(__name__)

#: The K in reciprocal rank fusion. 60 is the value from the original paper and
#: is not especially sensitive: it controls how sharply top ranks are favoured
#: over lower ones. Larger K flattens the curve.
RRF_K = 60

#: Per-ranker weights. Value matches are weighted highest because they are the
#: strongest evidence available: if the question literally contains a string
#: stored in a column, that column is almost certainly relevant. BM25 edges out
#: embeddings because schema vocabulary is unusually literal.
WEIGHT_BM25 = 1.0
WEIGHT_EMBEDDING = 0.8
WEIGHT_VALUE = 2.0

#: A table whose column was matched inherits some of that column's standing.
#: Without this, a question matching only ``is_defaulted`` would rank the
#: column highly while its table `loan` languished and got cut.
COLUMN_TO_TABLE_CREDIT = 0.6

#: Tables that are almost always worth including regardless of the question.
#: `account_summary` is here because balance questions are the single most
#: common category and the model must not answer them by scanning the
#: transaction table.
ALWAYS_CONSIDER: tuple[str, ...] = ("account_summary",)


@dataclass
class SchemaLinkResult:
    """What the linker decided, and enough evidence to see why."""

    tables: list[str] = field(default_factory=list)
    columns: dict[str, list[str]] = field(default_factory=dict)
    value_hits: list[ValueHit] = field(default_factory=list)

    #: Fused score per table, for debugging and for the API's trace payload.
    table_scores: dict[str, float] = field(default_factory=dict)

    #: "hybrid" normally, "all-tables" when retrieval is switched off.
    method: str = "hybrid"
    latency_ms: float = 0.0

    @property
    def column_count(self) -> int:
        return sum(len(c) for c in self.columns.values())

    def render_value_hints(self) -> str:
        """Prompt lines describing which literals were found where."""
        if not self.value_hits:
            return ""
        seen: set[str] = set()
        lines: list[str] = []
        for hit in self.value_hits:
            if hit.qualified_column in seen:
                continue
            seen.add(hit.qualified_column)
            lines.append(hit.render_hint())
        return "\n".join(lines)

    def summary(self) -> dict[str, object]:
        """Compact form for structured logs."""
        return {
            "method": self.method,
            "tables": ",".join(self.tables),
            "n_columns": self.column_count,
            "n_value_hits": len(self.value_hits),
            "latency_ms": round(self.latency_ms, 1),
        }


class SchemaLinker:
    """Builds the indexes once, then answers link queries cheaply."""

    def __init__(
        self,
        catalog: Catalog,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.catalog = catalog
        self.settings = settings or get_settings()

        documents = catalog.search_documents()
        self._doc_by_id = {d["id"]: d for d in documents}

        # -- lexical
        self._bm25 = BM25Index.build([(d["id"], d["text"]) for d in documents])

        # -- values
        self._values = ValueIndex.from_catalog(catalog)

        # -- dense
        self._embedder = embedder or get_embedder(
            self.settings.embedding_model,
            enabled=self.settings.embeddings_enabled,
        )
        cache = EmbeddingCache(self.settings.cache_dir)
        self._doc_ids = [d["id"] for d in documents]
        self._doc_vectors = cache.get_or_compute(
            self._embedder, [d["text"] for d in documents]
        )

        log.info(
            "schema_linker_ready",
            extra={
                "documents": len(documents),
                "values_indexed": self._values.size,
                "embedder": self._embedder.name,
            },
        )

    # -- individual rankers -------------------------------------------------
    def _rank_bm25(self, question: str, top_k: int) -> list[str]:
        return [doc_id for doc_id, _ in self._bm25.search(question, top_k=top_k)]

    def _rank_embeddings(self, question: str, top_k: int) -> list[str]:
        if self._doc_vectors.size == 0:
            return []
        query_vector = self._embedder.encode([question])[0]
        # Vectors are L2-normalised on both sides, so a dot product *is* the
        # cosine similarity. No division needed.
        similarities = self._doc_vectors @ query_vector
        # argpartition beats a full sort when we only want the top few, though
        # at 151 documents this is entirely theoretical.
        count = min(top_k, len(similarities))
        best = np.argpartition(-similarities, count - 1)[:count]
        best = best[np.argsort(-similarities[best])]
        return [self._doc_ids[i] for i in best]

    def _rank_values(self, hits: list[ValueHit]) -> list[str]:
        """Turn value hits into document ids, best evidence first."""
        ranked: list[str] = []
        for hit in hits:
            doc_id = f"column::{hit.table}.{hit.column}"
            if doc_id in self._doc_by_id and doc_id not in ranked:
                ranked.append(doc_id)
            table_doc = f"table::{hit.table}"
            if table_doc in self._doc_by_id and table_doc not in ranked:
                ranked.append(table_doc)
        return ranked

    # -- fusion -------------------------------------------------------------
    @staticmethod
    def _fuse(rankings: list[tuple[list[str], float]]) -> dict[str, float]:
        """Reciprocal rank fusion over several ranked lists."""
        scores: dict[str, float] = {}
        for ranked_ids, weight in rankings:
            for rank, doc_id in enumerate(ranked_ids, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (RRF_K + rank)
        return scores

    def link(
        self,
        question: str,
        *,
        top_k_tables: int | None = None,
        top_k_columns: int | None = None,
    ) -> SchemaLinkResult:
        """Select the tables and columns relevant to ``question``."""
        started = time.perf_counter()
        top_k_tables = top_k_tables or self.settings.retrieval_top_k_tables
        top_k_columns = top_k_columns or self.settings.retrieval_top_k_columns

        if not self.settings.retrieval_enabled:
            result = SchemaLinkResult(
                tables=self.catalog.table_names,
                columns={},                       # empty means "all columns"
                value_hits=self._values.search(question),
                method="all-tables",
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        # Retrieve generously, then trim. Asking each ranker for only the final
        # number would mean a document had to rank highly for *every* ranker to
        # survive, which defeats the point of fusing them.
        candidate_pool = max(top_k_columns, 40)

        value_hits = self._values.search(question)
        fused = self._fuse(
            [
                (self._rank_bm25(question, candidate_pool), WEIGHT_BM25),
                (self._rank_embeddings(question, candidate_pool), WEIGHT_EMBEDDING),
                (self._rank_values(value_hits), WEIGHT_VALUE),
            ]
        )

        # Roll column scores up into their tables. A table is relevant if its
        # own document scored, or if its columns did.
        table_scores: dict[str, float] = {}
        column_scores: dict[str, dict[str, float]] = {}

        for doc_id, score in fused.items():
            document = self._doc_by_id.get(doc_id)
            if document is None:
                continue
            table_name = document["table"]

            if document["kind"] == "table":
                table_scores[table_name] = table_scores.get(table_name, 0.0) + score
            else:
                table_scores[table_name] = (
                    table_scores.get(table_name, 0.0) + score * COLUMN_TO_TABLE_CREDIT
                )
                column_scores.setdefault(table_name, {})[document["column"]] = score

        for name in ALWAYS_CONSIDER:
            if name in self.catalog.tables:
                table_scores.setdefault(name, 0.0)

        chosen_tables = [
            name
            for name, _ in sorted(table_scores.items(), key=lambda kv: kv[1], reverse=True)
        ][:top_k_tables]

        # Pull in tables that the chosen ones point at via foreign keys.
        # Without this the model gets `loan` but not `account`, and cannot
        # write the join it needs. Cheap insurance against a broken query.
        chosen_tables = self._close_over_foreign_keys(chosen_tables, limit=top_k_tables + 2)

        # Distribute the column budget across the chosen tables, best-scoring
        # columns first, so a table nobody asked much about does not eat the
        # allowance of one at the centre of the question.
        columns = self._select_columns(chosen_tables, column_scores, top_k_columns)

        result = SchemaLinkResult(
            tables=chosen_tables,
            columns=columns,
            value_hits=value_hits,
            table_scores={k: round(v, 4) for k, v in sorted(
                table_scores.items(), key=lambda kv: kv[1], reverse=True)},
            method="hybrid",
            latency_ms=(time.perf_counter() - started) * 1000,
        )

        log.debug("schema_linked", extra=result.summary())
        return result

    def _close_over_foreign_keys(self, tables: list[str], *, limit: int) -> list[str]:
        """Add tables the chosen ones reference, so joins remain writable."""
        selected = list(tables)
        seen = set(selected)

        for name in tables:
            table = self.catalog.table(name)
            if table is None:
                continue
            for _, target_table, _ in table.foreign_keys:
                if target_table not in seen and len(selected) < limit:
                    selected.append(target_table)
                    seen.add(target_table)
        return selected

    def _select_columns(
        self,
        tables: list[str],
        column_scores: dict[str, dict[str, float]],
        budget: int,
    ) -> dict[str, list[str]]:
        """Choose which columns of each table to show, within a total budget."""
        result: dict[str, list[str]] = {}
        remaining = budget

        # Flatten to (table, column, score) and take globally best first, so
        # the budget follows the question rather than the table order.
        flattened: list[tuple[str, str, float]] = [
            (table, column, score)
            for table in tables
            for column, score in column_scores.get(table, {}).items()
        ]
        flattened.sort(key=lambda item: item[2], reverse=True)

        for table_name, column_name, _ in flattened:
            if remaining <= 0:
                break
            bucket = result.setdefault(table_name, [])
            if column_name not in bucket:
                bucket.append(column_name)
                remaining -= 1

        for table_name in tables:
            table = self.catalog.table(table_name)
            if table is None:
                continue
            # Under 12 columns: just show all of them.
            if len(table.columns) <= 12:
                result[table_name] = table.column_names
            else:
                result.setdefault(table_name, table.column_names[:8])

        return result
