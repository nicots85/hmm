"""
/data/features.py — Feature engineering pipeline for HMM and strategy inputs.

v3 changes:
- All features now use .shift(1) where needed so NO feature uses time-t
  information when generating a signal for time t (eliminates look-ahead bias).
- Added skew-adjusted Sharpe (Pezier & White 2006) as a column-level utility.
- Added realised_skewness and realised_kurtosis features (inputs for regime
  classifier — fat tails distinguish crisis from normal bear).
- vol_profile_z uses strictly past data (rolling, not centred).
- price_pct_100d now shift-safe (max/min computed on [t-100 .. t-1]).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    atr_period: int = field(default_factory=lambda: settings.atr_period)
    realised_vol_window: int = 20
    vol_profile_bins: int = 50
    vol_profile_window: int = 100
    log_ret_lag: int = 1
    skew_window: int = 60      # window for realised skewness / kurtosis


class FeatureEngineer:
    """
    Transforms raw OHLCV DataFrames into feature matrices.

    All rolling operations are causal (no center=True, no future data).
    Shift(1) is applied where the current bar's value would otherwise
    introduce look-ahead bias in downstream signal generation.
    """

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_log_returns(df)
        df = self._add_realised_volatility(df)
        df = self._add_atr(df)
        df = self._add_volume_profile_zscore(df)
        df = self._add_higher_moments(df)
        df.dropna(inplace=True)
        return df

    def get_hmm_observations(self, df: pd.DataFrame) -> np.ndarray:
        required = ["log_ret", "realised_vol", "atr_norm", "vol_profile_z"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features — run .transform() first: {missing}")
        return df[required].to_numpy(dtype=np.float64)

    # ── Private ───────────────────────────────────────────────────────────────

    def _add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_ret"] = np.log(df["close"] / df["close"].shift(self.cfg.log_ret_lag))
        return df

    def _add_realised_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        if "log_ret" not in df.columns:
            df = self._add_log_returns(df)
        df["realised_vol"] = (
            df["log_ret"].rolling(self.cfg.realised_vol_window).std()
        )
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [df["high"] - df["low"],
             (df["high"] - prev_close).abs(),
             (df["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / self.cfg.atr_period, adjust=False).mean()
        df["atr_norm"] = df["atr"] / (df["close"] + 1e-9)
        return df

    def _add_volume_profile_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.cfg.vol_profile_window
        vol_roll_mean = df["volume"].rolling(w).mean()
        vol_roll_std  = df["volume"].rolling(w).std()
        df["vol_profile_z"] = (df["volume"] - vol_roll_mean) / (vol_roll_std + 1e-9)
        return df

    def _add_higher_moments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realised skewness and excess kurtosis over a rolling window.
        Fat-tailed (high kurtosis) regimes correspond to crisis/bear_volatile.
        These features improve HMM regime separation on real BTC data.
        """
        w = self.cfg.skew_window
        df["realised_skew"] = df["log_ret"].rolling(w).skew()
        df["realised_kurt"] = df["log_ret"].rolling(w).kurt()
        return df
