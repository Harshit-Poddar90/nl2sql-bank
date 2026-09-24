"""Choosing which worked examples to show the model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from nl2sql.config import Settings, get_settings
from nl2sql.generation.prompts import CANNOT_ANSWER_MARKER
from nl2sql.logging_setup import get_logger
from nl2sql.retrieval.bm25 import BM25Index
from nl2sql.retrieval.embeddings import Embedder, EmbeddingCache, get_embedder

log = get_logger(__name__)

#: How much to penalise an example for resembling one already chosen.
#: 0.0 is plain top-k; 1.0 optimises purely for difference and starts returning
#: irrelevant examples. 0.3 keeps relevance dominant while breaking up clusters.
DIVERSITY_PENALTY = 0.3

#: Relative weight of lexical overlap against embedding similarity.
BM25_WEIGHT = 0.4
EMBEDDING_WEIGHT = 0.6


@dataclass(frozen=True)
class Example:
    """One worked question/SQL pair."""

    question: str
    sql: str
    tags: tuple[str, ...] = ()

    @property
    def is_refusal(self) -> bool:
        return CANNOT_ANSWER_MARKER in self.sql

    def as_pair(self) -> tuple[str, str]:
        return (self.question, self.sql)


def load_examples(path: Path) -> list[Example]:
    """Read the example bank from YAML."""
    if not path.exists():
        log.warning("example_bank_missing", extra={"path": str(path)})
        return []

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    examples: list[Example] = []
    for entry in raw.get("examples", []):
        question = (entry.get("question") or "").strip()
        sql = (entry.get("sql") or "").strip()
        if question and sql:
            examples.append(
                Example(question=question, sql=sql, tags=tuple(entry.get("tags", [])))
            )

    log.debug("example_bank_loaded", extra={"count": len(examples), "path": path.name})
    return examples


class FewShotSelector:
    """Picks the most useful examples for a given question."""

    def __init__(
        self,
        examples: list[Example],
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.examples = examples

        if not examples:
            self._bm25 = BM25Index.build([])
            self._vectors = np.zeros((0, 1), dtype=np.float32)
            self._embedder = embedder or get_embedder(enabled=False)
            return

        # Index on the question text plus its tags. Tags act as a hand-written
        # relevance signal: tagging an example 'default' makes it findable from
        # a question saying "defaulted" even if the wording differs completely.
        documents = [
            f"{example.question} {' '.join(example.tags)}" for example in examples
        ]
        self._bm25 = BM25Index.build(
            [(str(i), doc) for i, doc in enumerate(documents)]
        )

        self._embedder = embedder or get_embedder(
            self.settings.embedding_model, enabled=self.settings.embeddings_enabled
        )
        cache = EmbeddingCache(self.settings.cache_dir)
        self._vectors = cache.get_or_compute(self._embedder, documents)

        log.debug("few_shot_selector_ready", extra={"examples": len(examples)})

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, embedder: Embedder | None = None
    ) -> FewShotSelector:
        settings = settings or get_settings()
        return cls(load_examples(settings.few_shot_path), settings, embedder)

    def select(self, question: str, count: int | None = None) -> list[Example]:
        """Return the ``count`` most useful examples for ``question``."""
        count = self.settings.few_shot_count if count is None else count
        if count <= 0 or not self.examples:
            return []

        relevance = self._score(question)
        selected = self._select_diverse(relevance, count)

        # Make sure the model sees that refusing is an option.
        if not any(self.examples[i].is_refusal for i in selected):
            refusal = next(
                (i for i, e in enumerate(self.examples) if e.is_refusal), None
            )
            if refusal is not None and selected:
                selected[-1] = refusal      # replace the weakest pick
            elif refusal is not None:
                selected = [refusal]

        return [self.examples[i] for i in selected]

    def _score(self, question: str) -> np.ndarray:
        """Combined relevance of every example to the question, in [0, 1]-ish."""
        n = len(self.examples)
        scores = np.zeros(n, dtype=np.float32)

        # -- lexical, normalised so it can be mixed with cosine similarity
        bm25_hits = self._bm25.search(question, top_k=n)
        if bm25_hits:
            best = max(score for _, score in bm25_hits) or 1.0
            for doc_id, score in bm25_hits:
                scores[int(doc_id)] += BM25_WEIGHT * (score / best)

        # -- dense
        if self._vectors.size:
            query_vector = self._embedder.encode([question])[0]
            similarities = self._vectors @ query_vector
            scores += EMBEDDING_WEIGHT * similarities

        return scores

    def _select_diverse(self, relevance: np.ndarray, count: int) -> list[int]:
        """Maximal Marginal Relevance: relevant, but not all the same."""
        count = min(count, len(self.examples))
        chosen: list[int] = []
        remaining = set(range(len(self.examples)))

        has_vectors = self._vectors.size > 0

        while len(chosen) < count and remaining:
            best_index, best_score = None, -np.inf

            for index in remaining:
                score = float(relevance[index])
                if chosen and has_vectors:
                    similarity = float(
                        max(self._vectors[index] @ self._vectors[j] for j in chosen)
                    )
                    score -= DIVERSITY_PENALTY * similarity
                if score > best_score:
                    best_index, best_score = index, score

            if best_index is None:
                break
            chosen.append(best_index)
            remaining.discard(best_index)

        return chosen
