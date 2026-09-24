"""Dense embeddings for semantic schema matching, with a guaranteed fallback."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from nl2sql.logging_setup import get_logger

log = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into vectors."""

    name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch. Returns shape ``(len(texts), dimension)``."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows alone."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic character-n-gram hashing embedder. No dependencies, no downloads."""

    def __init__(self, dimension: int = 384) -> None:
        self.name = "hashing-ngram"
        self.dimension = dimension

    @staticmethod
    def _bucket(token: str, dimension: int) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row, text in enumerate(texts):
            cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                continue

            # Whole words carry more meaning than a sliding window, so they get
            # extra weight.
            for word in cleaned.split():
                matrix[row, self._bucket(f"w:{word}", self.dimension)] += 2.0

            padded = f" {cleaned} "
            for size in (3, 4, 5):
                for start in range(len(padded) - size + 1):
                    gram = padded[start : start + size]
                    matrix[row, self._bucket(gram, self.dimension)] += 1.0

        return _normalise(matrix)


class FastEmbedEmbedder:
    """Real sentence embeddings via fastembed + ONNX Runtime."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "fastembed is not installed. Install it with: pip install 'nl2sql-bank[embeddings]'"
            ) from exc

        log.info("loading_embedding_model", extra={"model": model_name})
        self._model = TextEmbedding(model_name=model_name)
        self.name = model_name

        # Ask the model rather than hard-coding 384: swapping to a larger model
        # via NL2SQL_EMBEDDING_MODEL should just work.
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dimension = int(np.asarray(probe).shape[-1])

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return _normalise(vectors)


def get_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    *,
    enabled: bool = True,
) -> Embedder:
    """Return the best embedder available, without ever failing."""
    if not enabled:
        log.info("embeddings_disabled", extra={"fallback": "hashing-ngram"})
        return HashingEmbedder()

    try:
        return FastEmbedEmbedder(model_name)
    except Exception as exc:
        log.warning(
            "embedding_model_unavailable",
            extra={"model": model_name, "error": str(exc), "fallback": "hashing-ngram"},
        )
        return HashingEmbedder()


class EmbeddingCache:
    """Disk cache for a corpus's embedding matrix."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name: str, texts: list[str]) -> Path:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        for text in texts:
            digest.update(b"\x00")
            digest.update(text.encode("utf-8"))
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        return self.cache_dir / f"emb-{safe_model}-{digest.hexdigest()[:16]}.npy"

    def get_or_compute(self, embedder: Embedder, texts: list[str]) -> np.ndarray:
        """Load the matrix from disk, or compute and store it."""
        path = self._path(embedder.name, texts)

        if path.exists():
            try:
                cached = np.load(path)
                if cached.shape[0] == len(texts):
                    log.debug("embedding_cache_hit", extra={"path": path.name})
                    return cached
            except Exception as exc:
                log.warning("embedding_cache_unreadable", extra={"error": str(exc)})

        vectors = embedder.encode(texts)

        try:
            # One cache file per (model, corpus). Drop older ones for this model
            # so the directory does not grow without bound as the schema evolves.
            safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", embedder.name)
            for stale in self.cache_dir.glob(f"emb-{safe_model}-*.npy"):
                stale.unlink(missing_ok=True)
            np.save(path, vectors)
        except Exception as exc:  # pragma: no cover
            log.warning("embedding_cache_write_failed", extra={"error": str(exc)})

        return vectors
