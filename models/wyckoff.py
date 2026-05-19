"""
/models/wyckoff.py — Algorithmic Wyckoff phase & event detection.

Design decisions:
- Wyckoff analysis is inherently multi-timeframe and subjective; this
  implementation operationalises it via four objective, measurable criteria:
    1. Price structure (swing high/low detection via rolling argmax/argmin).
    2. Volume Spread Analysis (VSA): bar spread × volume for effort/result.
    3. Effort vs Result divergence: large volume + small bar = absorption.
    4. Momentum proxy: rate-of-change over a configurable lookback.

- Phases are scored continuously (0–1 probability) rather than as hard
  labels. The strategy layer applies a threshold (default: 0.65) to treat
  the phase as confirmed.

- Spring and Upthrust are the two highest-signal events in Wyckoff:
  Spring: price briefly breaks support, then reverses upward (shakeout).
  Upthrust: price briefly breaks resistance, then reverses downward.
  Both are detected by checking for a wick that breaches the structural
  level followed by a close on the opposite side of the level.

- No proprietary data beyond OHLCV + volume is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WyckoffConfig:
    swing_window: int = 20          # bars for swing high/low detection
    vsa_spread_ma: int = 20         # MA window for spread normalisation
    volume_ma: int = 20             # MA window for relative volume
    roc_window: int = 10            # rate-of-change window for momentum
    absorption_ratio: float = 1.5   # volume > N×avg while spread < 0.3×avg
    phase_score_threshold: float = 0.65  # min score to confirm a phase
    wick_tolerance: float = 0.005   # relative tolerance for level breach


@dataclass
class WyckoffSnapshot:
    """Single-bar output from the Wyckoff analyser."""

    timestamp: pd.Timestamp
    phase: str                  # "accumulation" | "distribution" | "markup" | "markdown" | "ranging"
    phase_score: float          # 0–1 confidence
    spring_detected: bool
    upthrust_detected: bool
    absorption_score: float     # 0–1 proxy for institutional absorption
    effort_result_div: float    # negative = absorption, positive = expansion


class WyckoffAnalyser:
    """
    Identifies Wyckoff phases and key events (Spring, Upthrust) from OHLCV.

    Usage:
        wa = WyckoffAnalyser()
        enriched = wa.analyse(ohlcv_df)  # returns df with Wyckoff columns
        snapshot = wa.latest_snapshot(enriched)
    """

    def __init__(self, cfg: WyckoffConfig | None = None) -> None:
        self.cfg = cfg or WyckoffConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich OHLCV DataFrame with Wyckoff signals.

        Returns:
            df copy with columns:
            [swing_high, swing_low, spread, spread_ma, vol_ma, rel_vol,
             effort_result, absorption_score, phase, phase_score,
             spring, upthrust]
        """
        df = df.copy()
        df = self._add_swing_levels(df)
        df = self._add_vsa_metrics(df)
        df = self._add_phase(df)
        df = self._add_events(df)
        df.dropna(subset=["phase"], inplace=True)
        return df

    def latest_snapshot(self, enriched: pd.DataFrame) -> WyckoffSnapshot:
        """Extract a WyckoffSnapshot from the most recent bar."""
        row = enriched.iloc[-1]
        return WyckoffSnapshot(
            timestamp=enriched.index[-1],
            phase=str(row["phase"]),
            phase_score=float(row["phase_score"]),
            spring_detected=bool(row["spring"]),
            upthrust_detected=bool(row["upthrust"]),
            absorption_score=float(row["absorption_score"]),
            effort_result_div=float(row["effort_result"]),
        )

    # ── Private methods ───────────────────────────────────────────────────────

    def _add_swing_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling swing high/low: a swing high at bar t means close[t] is the
        maximum of the surrounding `swing_window` bars.
        """
        w = self.cfg.swing_window
        df["swing_high"] = df["high"].rolling(w, center=True).max()
        df["swing_low"] = df["low"].rolling(w, center=True).min()
        return df

    def _add_vsa_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VSA:
        - spread = high - low (bar range proxy for spread)
        - effort_result = sign(close - open) * spread / (volume + 1)
          → positive = price expanding on volume (result)
          → near-zero or negative = price stalling on volume (effort w/o result = absorption)
        - absorption_score: bars where volume > N×vol_ma and spread < 0.3×spread_ma
          are candidate absorption bars; score is a rolling fraction of such bars.
        """
        w = self.cfg.vsa_spread_ma
        v_w = self.cfg.volume_ma

        df["spread"] = df["high"] - df["low"]
        df["spread_ma"] = df["spread"].rolling(w).mean()
        df["vol_ma"] = df["volume"].rolling(v_w).mean()
        df["rel_vol"] = df["volume"] / (df["vol_ma"] + 1e-9)

        direction = np.sign(df["close"] - df["open"])
        df["effort_result"] = direction * df["spread"] / (df["volume"] + 1e-9)

        high_vol = df["rel_vol"] > self.cfg.absorption_ratio
        tight_spread = df["spread"] < 0.3 * df["spread_ma"]
        absorption_bar = (high_vol & tight_spread).astype(float)
        df["absorption_score"] = absorption_bar.rolling(w).mean()

        return df

    def _add_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase scoring heuristic (each component in [0, 1]):

        Accumulation proxy:
          - Price in lower quartile of rolling 100-bar range
          - Absorption score above median
          - Negative trend (recent 20-bar ROC < 0)
          → phase_score = mean of the three boolean conditions

        Distribution proxy:
          - Price in upper quartile of rolling 100-bar range
          - Absorption score above median
          - Positive trend

        Markup / Markdown: clear trend, low absorption.
        Ranging: no clear trend, low absorption.

        Hard labels are assigned to the regime with the highest score.
        """
        roc_w = self.cfg.roc_window
        df["roc"] = df["close"].pct_change(roc_w)

        range_high = df["close"].rolling(100).max()
        range_low = df["close"].rolling(100).min()
        range_width = range_high - range_low + 1e-9
        normalised_close = (df["close"] - range_low) / range_width  # 0 = bottom, 1 = top

        in_lower_quartile = (normalised_close < 0.25).astype(float)
        in_upper_quartile = (normalised_close > 0.75).astype(float)
        high_absorption = (df["absorption_score"] > df["absorption_score"].median()).astype(float)
        bearish_trend = (df["roc"] < 0).astype(float)
        bullish_trend = (df["roc"] > 0).astype(float)

        # Composite scores
        acc_score = (in_lower_quartile + high_absorption + bearish_trend) / 3.0
        dist_score = (in_upper_quartile + high_absorption + bullish_trend) / 3.0
        markup_score = bullish_trend * (1 - high_absorption)
        markdown_score = bearish_trend * (1 - high_absorption)
        ranging_score = ((df["roc"].abs() < 0.01).astype(float)) * (1 - high_absorption)

        scores = pd.DataFrame({
            "accumulation": acc_score,
            "distribution": dist_score,
            "markup": markup_score,
            "markdown": markdown_score,
            "ranging": ranging_score,
        }, index=df.index)

        df["phase"] = scores.idxmax(axis=1)
        df["phase_score"] = scores.max(axis=1)
        return df

    def _add_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Spring: a bar whose low temporarily dips below the rolling swing_low
        support but closes above it — shakeout of weak longs.

        Upthrust: a bar whose high temporarily exceeds the rolling swing_high
        resistance but closes below it — distribution into retail buyers.

        wick_tolerance allows for micro-breaches without false positives.
        """
        tol = self.cfg.wick_tolerance

        spring = (
            (df["low"] < df["swing_low"] * (1 - tol))
            & (df["close"] > df["swing_low"])
        )
        upthrust = (
            (df["high"] > df["swing_high"] * (1 + tol))
            & (df["close"] < df["swing_high"])
        )

        df["spring"] = spring
        df["upthrust"] = upthrust
        return df
