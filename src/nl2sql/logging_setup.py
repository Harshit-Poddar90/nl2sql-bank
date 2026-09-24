"""Logging configuration."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from nl2sql.config import LogFormat, get_settings

# Attributes the stdlib puts on every LogRecord. Anything *not* in this set was
# passed by us via `extra=` and is therefore a structured field worth emitting.
_STANDARD_RECORD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)

# Keys whose values must never reach a log file, in any format. Belt and
# braces: config never logs secrets in the first place, but a future caller
# might pass one by accident and this catches it.
_REDACTED_KEYS = frozenset({"api_key", "apikey", "authorization", "token", "password", "secret"})

_LEVEL_COLOURS = {
    "DEBUG": "\033[38;5;244m",   # grey
    "INFO": "\033[38;5;39m",     # blue
    "WARNING": "\033[38;5;214m", # amber
    "ERROR": "\033[38;5;203m",   # red
    "CRITICAL": "\033[48;5;203m\033[97m",
}
_RESET = "\033[0m"


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Pull the caller-supplied ``extra=`` fields off a record, redacting secrets."""
    fields: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _STANDARD_RECORD_ATTRS or key.startswith("_"):
            continue
        fields[key] = "***redacted***" if key.lower() in _REDACTED_KEYS else value
    return fields


class JSONFormatter(logging.Formatter):
    """One JSON object per line. Container-friendly."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # `default=str` so a stray Path, Decimal or datetime never crashes logging.
        return json.dumps(payload, default=str, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Aligned, colourised, key=value tail. For reading with your eyes."""

    def __init__(self, *, colour: bool = True) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "") if self.colour else ""
        reset = _RESET if colour else ""
        # Trim the shared `nl2sql.` prefix so the interesting part lines up.
        logger_name = record.name.removeprefix("nl2sql.")

        line = (
            f"{self.formatTime(record, self.datefmt)} "
            f"{colour}{record.levelname:<7}{reset} "
            f"{logger_name:<22} {record.getMessage()}"
        )
        fields = _extra_fields(record)
        if fields:
            rendered = " ".join(f"{k}={_render(v)}" for k, v in sorted(fields.items()))
            line += f"  {rendered}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _render(value: Any) -> str:
    """Compact single-line rendering of a field value."""
    if isinstance(value, float):
        return f"{value:.3g}"
    text = str(value)
    if len(text) > 120:
        text = text[:117] + "..."
    return text.replace("\n", "\\n")


_configured = False


def configure_logging(
    *,
    level: str | None = None,
    fmt: LogFormat | None = None,
    force: bool = False,
) -> None:
    """Install handlers on the ``nl2sql`` logger. Idempotent."""
    global _configured
    if _configured and not force:
        return

    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()
    resolved_fmt = fmt or settings.log_format

    # Logs go to stderr so that piping stdout (`nl2sql ask ... > out.json`)
    # gives you clean data instead of data mixed with diagnostics.
    handler = logging.StreamHandler(stream=sys.stderr)
    if resolved_fmt is LogFormat.JSON:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(HumanFormatter(colour=sys.stderr.isatty()))

    root = logging.getLogger("nl2sql")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)
    # Don't let records bubble to the root logger as well, or anything that
    # calls logging.basicConfig() (uvicorn, streamlit) prints them a second time.
    root.propagate = False

    # These are chatty at DEBUG and never tell us anything we want.
    for noisy in ("httpx", "httpcore", "urllib3", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger, configuring the subsystem on first use."""
    configure_logging()
    return logging.getLogger(name if name.startswith("nl2sql") else f"nl2sql.{name}")
