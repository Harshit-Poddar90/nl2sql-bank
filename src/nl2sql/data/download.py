"""Fetch the raw PKDD'99 Berka banking dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from nl2sql.config import Settings, get_settings
from nl2sql.exceptions import DatasetDownloadError
from nl2sql.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class SourceFile:
    """One CSV of the source dataset, and what we expect it to contain."""

    name: str
    expected_rows: int
    min_bytes: int

    @property
    def filename(self) -> str:
        return f"{self.name}.csv"


# The eight source tables, with their canonical row counts.
SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile("district", expected_rows=77, min_bytes=3_000),
    SourceFile("client", expected_rows=5_369, min_bytes=50_000),
    SourceFile("account", expected_rows=4_500, min_bytes=80_000),
    SourceFile("disp", expected_rows=5_369, min_bytes=60_000),
    SourceFile("card", expected_rows=892, min_bytes=15_000),
    SourceFile("loan", expected_rows=682, min_bytes=15_000),
    SourceFile("order", expected_rows=6_471, min_bytes=150_000),
    SourceFile("trans", expected_rows=1_056_320, min_bytes=50_000_000),
)

# Tried in order. Add your own here if these ever rot.
MIRRORS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/compfiggg-hu/berka-bank-cohort-analysis/main/data/raw",
)

# `trans.csv` is 68 MB, so give the whole download a generous budget while
# still capping how long we will sit on a stalled connection.
_TIMEOUT = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)
_CHUNK_BYTES = 1 << 20  # 1 MiB


def _sha256(path: Path) -> str:
    """Checksum a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_valid(path: Path, source: SourceFile) -> bool:
    """Is this file plausibly the CSV we wanted?"""
    if not path.exists():
        return False
    if path.stat().st_size < source.min_bytes:
        log.warning(
            "cached_file_too_small",
            extra={"file": source.filename, "bytes": path.stat().st_size,
                   "min_bytes": source.min_bytes},
        )
        return False
    # The Berka CSVs are semicolon-delimited with a quoted header. An HTML
    # error page will not start with '<' + a semicolon-bearing first line.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    if first_line.lstrip().startswith("<") or ";" not in first_line:
        log.warning("cached_file_not_csv", extra={"file": source.filename,
                                                  "head": first_line[:80]})
        return False
    return True


def _download_one(client: httpx.Client, source: SourceFile, destination: Path) -> None:
    """Stream a single CSV to disk, trying each mirror in turn."""
    temp_path = destination.with_suffix(".part")
    errors: list[str] = []

    for mirror in MIRRORS:
        url = f"{mirror.rstrip('/')}/{source.filename}"
        try:
            log.info("downloading", extra={"file": source.filename, "mirror": mirror})
            with client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                written = 0
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_bytes(_CHUNK_BYTES):
                        handle.write(chunk)
                        written += len(chunk)

            if written < source.min_bytes:
                errors.append(f"{mirror}: only {written} bytes (expected >= {source.min_bytes})")
                temp_path.unlink(missing_ok=True)
                continue

            temp_path.replace(destination)
            log.info(
                "downloaded",
                extra={"file": source.filename, "mb": round(written / 1e6, 1),
                       "sha256": _sha256(destination)[:12]},
            )
            return

        except httpx.HTTPError as exc:
            errors.append(f"{mirror}: {type(exc).__name__}: {exc}")
            temp_path.unlink(missing_ok=True)
            continue

    raise DatasetDownloadError(
        f"Could not download {source.filename} from any mirror.",
        user_message=(
            f"Failed to download {source.filename}. Check your internet connection, "
            f"or place the file manually in data/raw/."
        ),
        details={"attempts": errors},
    )


def download_dataset(
    settings: Settings | None = None,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Ensure every source CSV is present in ``data/raw``."""
    settings = settings or get_settings()
    destination_dir = settings.raw_data_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    to_fetch: list[SourceFile] = []

    for source in SOURCE_FILES:
        path = destination_dir / source.filename
        paths[source.name] = path
        if force or not _looks_valid(path, source):
            to_fetch.append(source)
        else:
            log.debug("using_cached", extra={"file": source.filename})

    if not to_fetch:
        log.info("dataset_ready", extra={"files": len(SOURCE_FILES), "source": "cache"})
        return paths

    total_mb = sum(s.min_bytes for s in to_fetch) / 1e6
    log.info(
        "download_starting",
        extra={"files": len(to_fetch), "approx_mb": round(total_mb, 1)},
    )

    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        for source in to_fetch:
            _download_one(client, source, destination_dir / source.filename)

    log.info("dataset_ready", extra={"files": len(SOURCE_FILES), "source": "download"})
    return paths

