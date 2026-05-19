"""
config.py — Centralised runtime settings via Pydantic-Settings.

All values are resolved from environment variables (populated from .env).
Import `settings` anywhere in the codebase; never read os.environ directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Exchange credentials ──────────────────────────────────────────────────
    binance_api_key: str = Field(default="", repr=False)
    binance_secret: str = Field(default="", repr=False)
    bybit_api_key: str = Field(default="", repr=False)
    bybit_secret: str = Field(default="", repr=False)
    oanda_api_key: str = Field(default="", repr=False)
    oanda_account_id: str = Field(default="", repr=False)
    oanda_environment: str = "practice"

    # ── Data sources ──────────────────────────────────────────────────────────
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    reuters_rss_feed: str = "https://feeds.reuters.com/reuters/businessNews"

    # ── Storage paths ─────────────────────────────────────────────────────────
    model_artifact_path: Path = Path("./models/artifacts")
    data_cache_path: Path = Path("./data/cache")

    # ── Runtime ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"

    # ── HMM hyperparameters ───────────────────────────────────────────────────
    hmm_n_components: int = 4
    hmm_covariance_type: str = "full"

    # ── Risk parameters ───────────────────────────────────────────────────────
    atr_period: int = 14
    kelly_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    max_position_risk_pct: float = Field(default=0.02, gt=0.0, le=0.1)
    hedge_correlation_window: int = 30

    @field_validator("hmm_covariance_type")
    @classmethod
    def _validate_cov_type(cls, v: str) -> str:
        allowed = {"spherical", "diag", "full", "tied"}
        if v not in allowed:
            raise ValueError(f"hmm_covariance_type must be one of {allowed}")
        return v

    def ensure_dirs(self) -> None:
        """Create storage directories if they do not exist."""
        self.model_artifact_path.mkdir(parents=True, exist_ok=True)
        self.data_cache_path.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere.
settings = Settings()
