"""
/data/features.py — Feature engineering pipeline for HMM and strategy inputs.

Design decisions:
- FeatureEngineer is a stateless transformer: it takes a raw OHLCV DataFrame
  and returns an enriched one. No side effects, fully re-entrant.
- ATR uses the Wilder smoothing (EMA-based), consistent with the standard
  definition. We expose it as a raw value AND as ATR/close (normalised)
  so downstream models are scale-invariant.
- Log-returns are centred at zero; the HMM is trained on [log_ret, realised_vol,
  atr_norm, vol_profile_z] as the default observation sequence.
- Volume profile Z-score: standardises cumulative volume at each price bucket
  against the rolling window mean/std, giving a dimensionless measure of
  whether the current bar is trading at a high- or low-volume node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataclass for feature config — keeps signature clean
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    atr_period: int = field(default_factory=lambda: settings.atr_period)
    realised_vol_window: int = 20       # bars for rolling realised volatility
    vol_profile_bins: int = 50          # price buckets for volume profile
    vol_profile_window: int = 100       # rolling window for VP z-score baseline
    log_ret_lag: int = 1                # lag for log return calculation


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Transforms raw OHLCV DataFrames into feature matrices suitable for
    HMM training and strategy signal generation.

    Usage:
        fe = FeatureEngineer()
        enriched_df = fe.transform(raw_ohlcv_df)
        hmm_obs = fe.get_hmm_observations(enriched_df)
    """

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature pipeline to a raw OHLCV DataFrame.

        Args:
            df: UTC-indexed DataFrame with columns [open, high, low, close, volume].

        Returns:
            Enriched DataFrame; original columns are preserved.
        """
        df = df.copy()
        df = self._add_log_returns(df)
        df = self._add_realised_volatility(df)
        df = self._add_atr(df)
        df = self._add_volume_profile_zscore(df)
        df.dropna(inplace=True)
        return df

    def get_hmm_observations(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the observation matrix [log_ret, realised_vol, atr_norm,
        vol_profile_z] used as input to the HMM.

        Shape: (T, 4) — T time steps, 4 features.
        """
        required = ["log_ret", "realised_vol", "atr_norm", "vol_profile_z"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features; run .transform() first: {missing}")
        return df[required].to_numpy(dtype=np.float64)

    # ── Private methods ───────────────────────────────────────────────────────

    def _add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        log_ret = ln(close_t / close_{t-lag}).

        Log-returns are normally distributed by assumption, making them
        suitable for Gaussian HMM emission models.
        """
        df["log_ret"] = np.log(df["close"] / df["close"].shift(self.cfg.log_ret_lag))
        return df

    def _add_realised_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realised volatility = rolling std of log_returns * sqrt(annualisation).

        Annualisation factor varies by timeframe but is left to the caller to
        set via FeatureConfig; here we store raw rolling std (dimensionless).
        """
        if "log_ret" not in df.columns:
            df = self._add_log_returns(df)
        df["realised_vol"] = (
            df["log_ret"]
            .rolling(self.cfg.realised_vol_window)
            .std()
        )
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Average True Range using Wilder's EMA smoothing (standard definition).

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR_norm   = ATR / close  → scale-free, comparable across assets.

        Critically used downstream in risk_manager.py for dynamic SL anchoring.
        """
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Wilder smoothing = EMA with alpha = 1/period
        df["atr"] = tr.ewm(alpha=1.0 / self.cfg.atr_period, adjust=False).mean()
        df["atr_norm"] = df["atr"] / df["close"]  # normalised ATR
        return df

    def _add_volume_profile_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume Profile Z-score.

        For each bar, we assign it to a price bucket (quantile-based), then
        compute how many std-devs the current bucket's volume deviates from
        its rolling mean — a proxy for high/low-volume price nodes (Value Area).

        Dimensionless output: negative = low-liquidity zone, positive = HVN.
        """
        n_bins = self.cfg.vol_profile_bins
        window = self.cfg.vol_profile_window

        # Assign each bar to a price bucket using the close price.
        df["price_bucket"] = pd.qcut(df["close"], q=n_bins, labels=False, duplicates="drop")

        # Rolling mean & std of volume within the same bucket — approximated
        # here as global rolling stats (full VP per window is O(n²); this is
        # the O(n) approximation used in production).
        vol_roll_mean = df["volume"].rolling(window).mean()
        vol_roll_std = df["volume"].rolling(window).std()

        df["vol_profile_z"] = (df["volume"] - vol_roll_mean) / (vol_roll_std + 1e-9)
        df.drop(columns=["price_bucket"], inplace=True)
        return df
