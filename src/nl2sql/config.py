"""Every tunable setting, typed and validated, in one place."""

from __future__ import annotations

import os
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# <root>/src/nl2sql/config.py -> the CLI finds the data from any working directory.
# NL2SQL_HOME overrides it for a non-editable install (the Docker image).
PROJECT_ROOT: Path = Path(os.environ.get("NL2SQL_HOME") or Path(__file__).resolve().parents[2])
PACKAGE_ROOT: Path = Path(__file__).resolve().parent


class LLMProvider(str, Enum):
    """Which service turns a question into SQL. ``stub`` is the offline rule-based baseline."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    STUB = "stub"


class LogFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


# Used when NL2SQL_LLM_MODEL is blank.
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-sonnet-5",
    LLMProvider.OLLAMA: "qwen2.5-coder:7b",
    LLMProvider.STUB: "rule-based-stub-v1",
}

# USD per million tokens (input, output), matched by model-name prefix. An
# estimate for the eval report, not billing; unknown models are priced at 0.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (0.80, 4.00),
}


def _under_root(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


class Settings(BaseSettings):
    """``max_result_rows`` reads ``NL2SQL_MAX_RESULT_ROWS``, and so on."""

    model_config = SettingsConfigDict(
        env_prefix="NL2SQL_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- LLM ---------------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.GEMINI
    llm_model: str = ""                                  # blank = provider default
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=1024, ge=64, le=32_000)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)  # transport errors, not bad SQL

    # Never logged; `safe_summary()` redacts them.
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # -- Database (SQLite; relative paths resolve against the project root) --
    db_path: str = "data/db/bank.sqlite"
    query_log_path: str = "data/db/query_log.sqlite"

    # -- Safety limits -----------------------------------------------------
    max_result_rows: int = Field(default=1000, ge=1, le=100_000)
    query_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_repair_attempts: int = Field(default=2, ge=0, le=5)
    max_join_tables: int = Field(default=8, ge=1, le=64)

    # -- Retrieval ---------------------------------------------------------
    retrieval_enabled: bool = True                       # False = whole schema (ablation)
    retrieval_top_k_tables: int = Field(default=6, ge=1, le=200)
    retrieval_top_k_columns: int = Field(default=60, ge=1, le=2000)
    embeddings_enabled: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    few_shot_count: int = Field(default=5, ge=0, le=32)

    # -- Service -----------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    log_format: LogFormat = LogFormat.TEXT
    log_level: str = "INFO"

    # -- Paths -------------------------------------------------------------
    @property
    def db_file(self) -> Path:
        return _under_root(self.db_path)

    @property
    def db_url(self) -> str:
        # as_posix() keeps the URL valid on Windows.
        return f"sqlite:///{self.db_file.as_posix()}"

    @property
    def query_log_file(self) -> Path:
        return _under_root(self.query_log_path)

    @property
    def raw_data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def cache_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "cache"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "evaluation" / "reports"

    @property
    def semantics_path(self) -> Path:
        return PACKAGE_ROOT / "catalog" / "semantics.yaml"

    @property
    def schema_sql_path(self) -> Path:
        return PACKAGE_ROOT / "data" / "schema.sql"

    @property
    def few_shot_path(self) -> Path:
        return PACKAGE_ROOT / "generation" / "examples.yaml"

    @property
    def benchmark_path(self) -> Path:
        return PACKAGE_ROOT / "evaluation" / "benchmark.jsonl"

    # -- Validators --------------------------------------------------------
    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, v: object) -> object:
        """Accept ' Gemini ', 'claude', 'offline' etc. Blank means the default."""
        if isinstance(v, str):
            cleaned = v.strip().lower() or "gemini"
            aliases = {"google": "gemini", "claude": "anthropic", "gpt": "openai",
                       "offline": "stub", "local": "ollama"}
            return aliases.get(cleaned, cleaned)
        return v

    @field_validator("llm_model", "gemini_api_key", "openai_api_key", "anthropic_api_key",
                     mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        """Pasting a key from a browser very often brings a newline."""
        return v.strip() if isinstance(v, str) else v

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, v: object) -> object:
        if isinstance(v, str):
            level = v.strip().upper() or "INFO"
            if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
                raise ValueError(f"log_level must be DEBUG/INFO/WARNING/ERROR/CRITICAL, got {v!r}")
            return level
        return v

    # -- Convenience -------------------------------------------------------
    @property
    def resolved_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]

    @property
    def active_api_key(self) -> str:
        """The key for the selected provider (empty for stub/ollama)."""
        return {
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
        }.get(self.llm_provider, "")

    @property
    def requires_api_key(self) -> bool:
        return self.llm_provider not in {LLMProvider.OLLAMA, LLMProvider.STUB}

    def price_per_mtok(self) -> tuple[float, float]:
        """(input, output) USD per million tokens for the active model; (0, 0) if unknown."""
        model = self.resolved_model
        return next((p for prefix, p in MODEL_PRICING_USD_PER_MTOK.items()
                     if model.startswith(prefix)), (0.0, 0.0))

    def ensure_directories(self) -> None:
        for directory in (self.raw_data_dir, self.db_file.parent, self.cache_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def safe_summary(self) -> dict[str, object]:
        """Config with secrets redacted -- safe to print."""
        key = self.active_api_key
        return {
            "llm_provider": self.llm_provider.value,
            "llm_model": self.resolved_model,
            "llm_temperature": self.llm_temperature,
            "api_key_present": bool(key),
            "api_key_hint": f"...{key[-4:]}" if len(key) >= 4 else None,
            "db_path": str(self.db_file),
            "max_result_rows": self.max_result_rows,
            "query_timeout_seconds": self.query_timeout_seconds,
            "max_repair_attempts": self.max_repair_attempts,
            "retrieval_enabled": self.retrieval_enabled,
            "embeddings_enabled": self.embeddings_enabled,
            "few_shot_count": self.few_shot_count,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings. Exits with a readable message on a bad ``.env``."""
    try:
        settings = Settings()
    except Exception as exc:
        print(f"\n[nl2sql] Configuration error:\n\n{exc}\n", file=sys.stderr)
        print("Check your .env file against .env.example.\n", file=sys.stderr)
        raise SystemExit(2) from exc
    settings.ensure_directories()
    return settings
