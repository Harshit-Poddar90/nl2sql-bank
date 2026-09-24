"""The semantic catalog: what the model is allowed to know about the database."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import Engine, create_engine, inspect, text

from nl2sql.config import Settings, get_settings
from nl2sql.exceptions import CatalogError, DataError
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)

#: A string column with at most this many distinct values is treated as an
#: enum: every value is listed in the prompt, so the model can pick the right
#: literal instead of inventing plausible-looking English.
ENUM_MAX_DISTINCT = 40

#: A column with at most this many distinct values gets all of them recorded
#: for the *value index*, which is a retrieval structure and never rendered
#: into a prompt.
#:
#: The two limits are separate on purpose. `district_name` has 77 values:
#: far too many to list in every prompt, but we absolutely want a question
#: mentioning "Benesov" to link to that column. Conflating the two would force
#: a choice between a bloated prompt and a retriever that cannot see place
#: names.
VALUE_INDEX_MAX_DISTINCT = 200

#: Sample values kept for high-cardinality columns. Enough to convey the format
#: (is this an ISO date or a YYMMDD integer?), few enough to stay cheap.
SAMPLE_VALUE_COUNT = 5

#: Bump when the catalog's on-disk JSON layout changes, so old caches are
#: ignored instead of being deserialised into the wrong shape.
CACHE_VERSION = 4


@dataclass
class ColumnInfo:
    """One column, as the model will see it."""

    name: str
    table: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False

    #: Human description, from semantics.yaml.
    description: str = ""
    #: Alternative words a user might say for this column.
    synonyms: list[str] = field(default_factory=list)

    #: Target of this column's foreign key, as "table.column", if any.
    references: str | None = None

    #: Every distinct value, for enum-like columns. Rendered into prompts.
    enum_values: list[str] = field(default_factory=list)
    #: Every distinct value for any column below VALUE_INDEX_MAX_DISTINCT.
    #: Used by the value index only; never rendered into a prompt.
    searchable_values: list[str] = field(default_factory=list)
    #: A few real values, for everything else.
    sample_values: list[str] = field(default_factory=list)
    #: Range summary for numeric and date columns.
    value_range: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.name}"

    def render(self) -> str:
        """One line of DDL-with-commentary, as it appears in the prompt."""
        parts = [f"  {self.name} {self.data_type}"]
        if self.is_primary_key:
            parts.append("PRIMARY KEY")
        if self.references:
            # Real DDL spelling -- `REFERENCES account(account_id)`, not
            # `REFERENCES account.account_id`. Models pattern-match hard on
            # well-formed SQL, and half-correct DDL invites half-correct output.
            target_table, _, target_column = self.references.partition(".")
            parts.append(f"REFERENCES {target_table}({target_column})")

        line = " ".join(parts)

        notes: list[str] = []
        if self.description:
            notes.append(self.description.rstrip("."))
        if self.enum_values:
            rendered = ", ".join(f"'{v}'" for v in self.enum_values[:ENUM_MAX_DISTINCT])
            notes.append(f"one of: {rendered}")
        elif self.sample_values:
            rendered = ", ".join(f"'{v}'" for v in self.sample_values[:3])
            notes.append(f"e.g. {rendered}")
        elif self.value_range and not self._range_is_noise:
            low, high = self.value_range.get("min"), self.value_range.get("max")
            if low is not None and high is not None:
                notes.append(f"range {low} to {high}")

        return f"{line},  -- {'. '.join(notes)}" if notes else f"{line},"

    @property
    def _range_is_noise(self) -> bool:
        """True for columns whose numeric range tells the model nothing."""
        if self.is_primary_key or self.references:
            return True
        if self.name.endswith("_id"):
            return True
        # A 0-1 range is a boolean flag; the description already explains it.
        return self.value_range.get("min") == 0 and self.value_range.get("max") == 1

    def search_text(self) -> str:
        """Everything about this column worth matching a question against."""
        pieces = [
            self.name,
            self.name.replace("_", " "),
            self.table.replace("_", " "),
            self.description,
            " ".join(self.synonyms),
            # Values go into the searchable text too, so BM25 can match a
            # question's literal ("Prague", "gold") straight to the column
            # holding it. Capped so a 200-value column does not dominate the
            # length normalisation and starve shorter documents.
            " ".join(self.searchable_values[:80]),
        ]
        return " ".join(p for p in pieces if p)


@dataclass
class TableInfo:
    """One table or view, with its columns and its meaning."""

    name: str
    kind: str = "table"                     # "table" or "view"
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    row_count: int = 0
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)

    #: (local_column, target_table, target_column) for each foreign key.
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)

    #: Free-text guidance: gotchas, preferred join paths, "prefer X over Y".
    notes: list[str] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnInfo | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)

    def render(self, *, include_columns: list[str] | None = None) -> str:
        """Render as annotated DDL for the prompt."""
        if include_columns is None:
            chosen = self.columns
        else:
            wanted = {c.lower() for c in include_columns}
            wanted.update(k.lower() for k in self.primary_key)
            wanted.update(fk[0].lower() for fk in self.foreign_keys)
            chosen = [c for c in self.columns if c.name.lower() in wanted]
            if not chosen:
                chosen = self.columns

        header = f"CREATE {'VIEW' if self.kind == 'view' else 'TABLE'} {self.name} ("
        lines = [header]
        lines.extend(column.render() for column in chosen)

        # Trim the trailing comma from the last column line so the DDL is
        # syntactically plausible. Models pattern-match on well-formed SQL.
        if len(lines) > 1:
            last = lines[-1]
            if ",  --" in last:
                lines[-1] = last.replace(",  --", "  --", 1)
            elif last.endswith(","):
                lines[-1] = last[:-1]

        lines.append(");")

        preamble: list[str] = []
        if self.description:
            preamble.append(f"-- {self.description}")
        preamble.append(f"-- {self.row_count:,} rows")
        if include_columns is not None and len(chosen) < len(self.columns):
            preamble.append(f"-- showing {len(chosen)} of {len(self.columns)} columns")
        for note in self.notes:
            preamble.append(f"-- NOTE: {note}")

        return "\n".join(preamble + lines)

    def search_text(self) -> str:
        pieces = [
            self.name,
            self.name.replace("_", " "),
            self.description,
            " ".join(self.synonyms),
            " ".join(self.notes),
            " ".join(c.name.replace("_", " ") for c in self.columns),
        ]
        return " ".join(p for p in pieces if p)


@dataclass
class Catalog:
    """The complete picture of the database, ready for retrieval and prompting."""

    tables: dict[str, TableInfo] = field(default_factory=dict)

    #: Business terms -> what they mean in this schema. Injected into prompts.
    glossary: dict[str, str] = field(default_factory=dict)

    #: Free-text rules that apply across the whole schema.
    global_notes: list[str] = field(default_factory=list)

    #: Canonical join paths, e.g. "client -> account" spelled out as SQL.
    join_hints: list[str] = field(default_factory=list)

    built_at: float = field(default_factory=time.time)

    # -- lookup ------------------------------------------------------------
    @property
    def table_names(self) -> list[str]:
        return sorted(self.tables)

    @property
    def base_table_names(self) -> list[str]:
        return sorted(n for n, t in self.tables.items() if t.kind == "table")

    def table(self, name: str) -> TableInfo | None:
        """Case-insensitive table lookup."""
        direct = self.tables.get(name)
        if direct is not None:
            return direct
        lowered = name.lower()
        return next((t for n, t in self.tables.items() if n.lower() == lowered), None)

    def all_columns(self) -> list[ColumnInfo]:
        return [column for table in self.tables.values() for column in table.columns]

    def identifier_allowlist(self) -> tuple[set[str], set[str]]:
        """The names the guardrail layer will accept."""
        tables = {name.lower() for name in self.tables}
        columns = {column.name.lower() for column in self.all_columns()}
        return tables, columns

    # -- prompt rendering ---------------------------------------------------
    def render_schema(
        self,
        table_names: list[str] | None = None,
        column_names: dict[str, list[str]] | None = None,
        *,
        include_glossary: bool = True,
        include_joins: bool = True,
    ) -> str:
        """Render the schema section of the prompt."""
        chosen = table_names or self.table_names
        blocks: list[str] = []

        for name in chosen:
            table = self.table(name)
            if table is None:
                continue
            per_table_columns = (column_names or {}).get(name)
            blocks.append(table.render(include_columns=per_table_columns))

        sections = ["\n\n".join(blocks)]

        if include_joins and self.join_hints:
            sections.append(
                "-- How these tables connect:\n"
                + "\n".join(f"--   {hint}" for hint in self.join_hints)
            )

        if include_glossary and self.glossary:
            # Only glossary entries relevant to the chosen tables, so a
            # three-table question does not carry the whole business dictionary.
            relevant = self._relevant_glossary(chosen)
            if relevant:
                sections.append(
                    "-- Business terms:\n"
                    + "\n".join(f"--   {term}: {meaning}" for term, meaning in relevant.items())
                )

        if self.global_notes:
            sections.append(
                "-- Rules for this database:\n"
                + "\n".join(f"--   {note}" for note in self.global_notes)
            )

        return "\n\n".join(section for section in sections if section.strip())

    def _relevant_glossary(self, table_names: list[str]) -> dict[str, str]:
        """Glossary entries that mention one of the chosen tables, plus universal ones."""
        chosen = {name.lower() for name in table_names}
        all_tables = {name.lower() for name in self.tables}
        result: dict[str, str] = {}

        for term, meaning in self.glossary.items():
            text_lower = f"{term} {meaning}".lower()
            mentioned = {t for t in all_tables if t in text_lower}
            if not mentioned or (mentioned & chosen):
                result[term] = meaning
        return result

    def search_documents(self) -> list[dict[str, Any]]:
        """The corpus the retriever indexes."""
        documents: list[dict[str, Any]] = []
        for table in self.tables.values():
            documents.append(
                {
                    "id": f"table::{table.name}",
                    "kind": "table",
                    "table": table.name,
                    "column": None,
                    "text": table.search_text(),
                }
            )
            for column in table.columns:
                documents.append(
                    {
                        "id": f"column::{column.qualified_name}",
                        "kind": "column",
                        "table": table.name,
                        "column": column.name,
                        "text": column.search_text(),
                    }
                )
        return documents

    def value_index_entries(self) -> list[tuple[str, str, str]]:
        """``(value, table, column)`` for every indexable value in the schema."""
        index: list[tuple[str, str, str]] = []
        for table in self.tables.values():
            for column in table.columns:
                for value in column.searchable_values:
                    # One-character values are noise: they match everywhere and
                    # mean nothing. The loan status codes 'A'..'D' are the
                    # deliberate exception -- questions do say "status A".
                    if value and (len(value) > 1 or column.name.endswith("_code")):
                        index.append((value, table.name, column.name))
        return index

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_version": CACHE_VERSION,
            "built_at": self.built_at,
            "glossary": self.glossary,
            "global_notes": self.global_notes,
            "join_hints": self.join_hints,
            "tables": {
                name: {
                    "name": t.name,
                    "kind": t.kind,
                    "description": t.description,
                    "synonyms": t.synonyms,
                    "row_count": t.row_count,
                    "primary_key": t.primary_key,
                    "foreign_keys": [list(fk) for fk in t.foreign_keys],
                    "notes": t.notes,
                    "columns": [
                        {
                            "name": c.name, "table": c.table, "data_type": c.data_type,
                            "nullable": c.nullable, "is_primary_key": c.is_primary_key,
                            "description": c.description, "synonyms": c.synonyms,
                            "references": c.references, "enum_values": c.enum_values,
                            "searchable_values": c.searchable_values,
                            "sample_values": c.sample_values, "value_range": c.value_range,
                        }
                        for c in t.columns
                    ],
                }
                for name, t in self.tables.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Catalog:
        catalog = cls(
            built_at=payload.get("built_at", time.time()),
            glossary=payload.get("glossary", {}),
            global_notes=payload.get("global_notes", []),
            join_hints=payload.get("join_hints", []),
        )
        for name, raw in payload.get("tables", {}).items():
            table = TableInfo(
                name=raw["name"],
                kind=raw.get("kind", "table"),
                description=raw.get("description", ""),
                synonyms=raw.get("synonyms", []),
                row_count=raw.get("row_count", 0),
                primary_key=raw.get("primary_key", []),
                foreign_keys=[tuple(fk) for fk in raw.get("foreign_keys", [])],
                notes=raw.get("notes", []),
            )
            table.columns = [ColumnInfo(**c) for c in raw.get("columns", [])]
            catalog.tables[name] = table
        return catalog


# ===========================================================================
# Building the catalog
# ===========================================================================

def _profile_column(
    engine: Engine,
    table_name: str,
    column: ColumnInfo,
    row_count: int,
) -> None:
    """Read real values out of the column and attach them to ``column``."""
    type_name = column.data_type.upper()
    is_text = any(token in type_name for token in ("CHAR", "TEXT", "CLOB"))
    is_numeric = any(token in type_name for token in ("INT", "REAL", "NUMER", "DEC", "FLOAT"))

    try:
        with engine.connect() as connection:
            if is_text:
                rows = connection.execute(
                    text(
                        f'SELECT DISTINCT "{column.name}" FROM "{table_name}" '
                        f'WHERE "{column.name}" IS NOT NULL '
                        f"LIMIT {VALUE_INDEX_MAX_DISTINCT + 1}"
                    )
                ).fetchall()
                # Sorted for stable output: an unstable prompt breaks provider
                # caching and makes benchmark runs non-reproducible.
                values = sorted({str(r[0]) for r in rows if r[0] is not None})

                if len(values) <= VALUE_INDEX_MAX_DISTINCT:
                    column.searchable_values = values
                if len(values) <= ENUM_MAX_DISTINCT:
                    column.enum_values = values
                else:
                    column.sample_values = values[:SAMPLE_VALUE_COUNT]

            elif is_numeric and row_count:
                row = connection.execute(
                    text(
                        f'SELECT MIN("{column.name}"), MAX("{column.name}") '
                        f'FROM "{table_name}"'
                    )
                ).fetchone()
                if row and row[0] is not None:
                    low, high = row
                    column.value_range = {
                        "min": round(low, 2) if isinstance(low, float) else low,
                        "max": round(high, 2) if isinstance(high, float) else high,
                    }
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("column_profile_failed",
                  extra={"table": table_name, "column": column.name, "error": str(exc)})


def introspect(engine: Engine, *, profile: bool = True) -> dict[str, TableInfo]:
    """Read the live database structure."""
    inspector = inspect(engine)
    tables: dict[str, TableInfo] = {}

    table_names = list(inspector.get_table_names())
    view_names = list(inspector.get_view_names())

    for name in table_names + view_names:
        kind = "view" if name in view_names else "table"

        try:
            primary_key = list(inspector.get_pk_constraint(name).get("constrained_columns") or [])
        except Exception:
            primary_key = []

        foreign_keys: list[tuple[str, str, str]] = []
        try:
            for fk in inspector.get_foreign_keys(name):
                referred = fk.get("referred_table")
                for local, remote in zip(
                    fk.get("constrained_columns") or [],
                    fk.get("referred_columns") or [],
                    strict=False,
                ):
                    if referred:
                        foreign_keys.append((local, referred, remote))
        except Exception:
            pass

        info = TableInfo(name=name, kind=kind, primary_key=primary_key,
                         foreign_keys=foreign_keys)

        fk_by_column = {local: f"{tbl}.{col}" for local, tbl, col in foreign_keys}

        for raw_column in inspector.get_columns(name):
            info.columns.append(
                ColumnInfo(
                    name=raw_column["name"],
                    table=name,
                    data_type=str(raw_column["type"]),
                    nullable=bool(raw_column.get("nullable", True)),
                    is_primary_key=raw_column["name"] in primary_key,
                    references=fk_by_column.get(raw_column["name"]),
                )
            )

        try:
            with engine.connect() as connection:
                info.row_count = int(
                    connection.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar_one()
                )
        except Exception:
            info.row_count = 0

        if profile:
            for column in info.columns:
                _profile_column(engine, name, column, info.row_count)

        tables[name] = info

    return tables


def _apply_semantics(tables: dict[str, TableInfo], semantics: dict[str, Any]) -> Catalog:
    """Layer the human-written meaning on top of the introspected structure."""
    catalog = Catalog(
        tables=tables,
        glossary=semantics.get("glossary", {}) or {},
        global_notes=semantics.get("global_notes", []) or [],
        join_hints=semantics.get("join_hints", []) or [],
    )

    problems: list[str] = []

    for table_name, spec in (semantics.get("tables", {}) or {}).items():
        table = catalog.table(table_name)
        if table is None:
            problems.append(f"semantics.yaml describes unknown table '{table_name}'")
            continue

        table.description = spec.get("description", "") or ""
        table.synonyms = spec.get("synonyms", []) or []
        table.notes = spec.get("notes", []) or []

        for column_name, column_spec in (spec.get("columns", {}) or {}).items():
            column = table.column(column_name)
            if column is None:
                problems.append(
                    f"semantics.yaml describes unknown column '{table_name}.{column_name}'"
                )
                continue

            if isinstance(column_spec, str):
                # Shorthand: `column_name: "its description"`.
                column.description = column_spec
            else:
                column.description = column_spec.get("description", "") or ""
                column.synonyms = column_spec.get("synonyms", []) or []
                # An explicit `values:` mapping overrides what profiling found.
                # Used where the raw values need explaining, not just listing.
                override = column_spec.get("values")
                if override:
                    column.enum_values = [str(v) for v in override]

    if problems:
        raise CatalogError(
            "The semantic catalog does not match the database.",
            user_message="semantics.yaml is out of date. See details.",
            details={"problems": problems},
        )

    return catalog


def _cache_key(settings: Settings) -> str:
    """Fingerprint the inputs, so a stale cache is never reused."""
    parts: list[str] = [str(CACHE_VERSION), settings.db_url]

    if settings.db_file.exists():
        stat = settings.db_file.stat()
        parts.append(f"{stat.st_size}:{int(stat.st_mtime)}")

    semantics = settings.semantics_path
    if semantics.exists():
        parts.append(str(int(semantics.stat().st_mtime)))

    import hashlib

    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build_catalog(
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
    profile: bool = True,
) -> Catalog:
    """Build (or load from cache) the semantic catalog."""
    settings = settings or get_settings()

    if not settings.db_file.exists():
        raise DataError(
            f"No warehouse at {settings.db_file}.",
            user_message="The database has not been built yet. Run: nl2sql data build",
        )

    cache_path = settings.cache_dir / f"catalog-{_cache_key(settings)}.json"
    if use_cache and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("cache_version") == CACHE_VERSION:
                log.debug("catalog_cache_hit", extra={"path": str(cache_path)})
                return Catalog.from_dict(payload)
        except Exception as exc:
            log.warning("catalog_cache_unreadable", extra={"error": str(exc)})

    started = time.perf_counter()
    engine = create_engine(settings.db_url, future=True)
    try:
        tables = introspect(engine, profile=profile)
    finally:
        engine.dispose()

    semantics = load_semantics(settings.semantics_path)
    catalog = _apply_semantics(tables, semantics)

    if use_cache:
        try:
            # Clear older catalogs so the cache directory does not accumulate
            # one file per rebuild forever.
            for stale in settings.cache_dir.glob("catalog-*.json"):
                stale.unlink(missing_ok=True)
            cache_path.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            log.warning("catalog_cache_write_failed", extra={"error": str(exc)})

    log.info(
        "catalog_built",
        extra={
            "tables": len(catalog.tables),
            "columns": len(catalog.all_columns()),
            "seconds": round(time.perf_counter() - started, 2),
        },
    )
    return catalog


def load_semantics(path: Path) -> dict[str, Any]:
    """Read semantics.yaml, tolerating its absence."""
    if not path.exists():
        log.warning("semantics_missing", extra={"path": str(path)})
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CatalogError(
            f"semantics.yaml is not valid YAML: {exc}",
            user_message="The semantic catalog file could not be parsed.",
        ) from exc
    return loaded or {}

