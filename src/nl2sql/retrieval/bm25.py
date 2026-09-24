"""BM25 lexical search over the schema."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Words that appear in nearly every question and so carry no signal about which
# table is meant. Kept deliberately short: over-aggressive stopword removal
# destroys real queries ("accounts with no card" hinges on "no").
STOPWORDS: frozenset[str] = frozenset(
    ["a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "from", "by", "with", "is", "are", "was", "were", "be", "been", "being", "how", "what", "which", "who", "whom", "whose", "when", "where", "why", "do", "does", "did", "done", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "have", "has", "had", "having", "i", "we", "you", "they", "it", "me", "my", "our", "your", "their", "this", "that", "these", "those", "there", "here", "as", "if", "then", "than", "show", "tell", "give", "list", "find", "get", "me", "please"]
)

# BM25 tuning. These are the standard values from the literature and there is
# no reason to deviate: k1 controls how quickly repeated terms stop helping,
# b controls how much a long document is penalised for its length.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Split text into comparable tokens."""
    lowered = text.lower()
    raw_tokens = re.findall(r"[a-z0-9]+", lowered)

    tokens: list[str] = []
    for token in raw_tokens:
        if token in STOPWORDS or len(token) < 2:
            continue
        # Depluralise, but never down to a stub: "is" must not become "i".
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


@dataclass
class BM25Index:
    """An in-memory BM25 index over a small set of documents."""

    #: Caller-supplied ids, parallel to the document list.
    doc_ids: list[str] = field(default_factory=list)
    #: Term frequencies per document.
    doc_terms: list[Counter[str]] = field(default_factory=list)
    #: Token count per document, for the length normalisation.
    doc_lengths: list[int] = field(default_factory=list)
    #: term -> number of documents containing it.
    document_frequency: Counter[str] = field(default_factory=Counter)
    average_length: float = 0.0

    @classmethod
    def build(cls, documents: list[tuple[str, str]]) -> BM25Index:
        """Index ``(doc_id, text)`` pairs."""
        index = cls()
        for doc_id, text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            index.doc_ids.append(doc_id)
            index.doc_terms.append(counts)
            index.doc_lengths.append(len(tokens))
            # Each distinct term in a document contributes 1 to its df.
            index.document_frequency.update(counts.keys())

        total = sum(index.doc_lengths)
        index.average_length = total / len(index.doc_lengths) if index.doc_lengths else 0.0
        return index

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def _idf(self, term: str) -> float:
        """Inverse document frequency, with the standard BM25 smoothing."""
        n = self.size
        df = self.document_frequency.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[str, float]]:
        """Score every document against ``query``."""
        query_terms = tokenize(query)
        if not query_terms or self.size == 0:
            return []

        scores: list[float] = [0.0] * self.size

        for term in set(query_terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, term_counts in enumerate(self.doc_terms):
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                length_norm = (
                    1 - B + B * (self.doc_lengths[i] / self.average_length)
                    if self.average_length
                    else 1.0
                )
                scores[i] += idf * (frequency * (K1 + 1)) / (frequency + K1 * length_norm)

        ranked = [
            (self.doc_ids[i], score) for i, score in enumerate(scores) if score > 0
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
