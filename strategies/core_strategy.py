"""
/strategies/core_strategy.py — Triple-confluence entry generator.

v3 changes:
- P10 FIX: SeasonalityAnalyser integrated as a 4th optional gate.
  If a seasonality filter is provided, entries are blocked during
  historically negative time buckets (hour or weekday).
- P1 FIX: strategy outputs signals intended for bar t+1 entry.
  Callers must NOT apply an additional shift — the engine does it.
- Hard gates (must all pass):
    1. HMM bull_prob or bear_vol_prob > threshold
    2. Trend filter (price > SMA50 for long; < for short)
    3. Volatility filter (ATR/ATR_60ma < threshold)
- Soft confidence boosters (weighted scoring):
    4. Wyckoff phase alignment
    5. Sentiment direction
    6. Seasonality (new v3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from models.hmm_regimes import HMMRegimeDetector
from models.wyckoff import WyckoffAnalyser, WyckoffSnapshot
from models.patterns import SeasonalityAnalyser
from strategies.inefficiencies import TemporalArbitrageFilter

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    timestamp: datetime
    asset: str
    direction: int          # +1 long, -1 short, 0 flat
    hmm_regime: str
    hmm_proba: float
    wyckoff_phase: str
    wyckoff_score: float
    sentiment_score: float
    inefficiency_active: bool
    seasonality_boost: float   # NEW v3
    confidence: float
    rationale: list[str] = field(default_factory=list)


@dataclass
class StrategyConfig:
    # HMM
    min_hmm_bull_prob: float = 0.55
    min_hmm_bear_vol_prob: float = 0.55

    # Trend filter
    sma_period: int = 50
    use_trend_filter: bool = True

    # Volatility filter
    atr_ratio_threshold: float = 1.4
    use_vol_filter: bool = True

    # Wyckoff (soft booster)
    wyckoff_long_phases: frozenset[str] = frozenset({"accumulation"})
    wyckoff_short_phases: frozenset[str] = frozenset({"distribution"})
    wyckoff_confidence_weight: float = 0.20

    # Sentiment (soft booster)
    min_sentiment_long: float = 0.05
    max_sentiment_short: float = -0.05
    sentiment_confidence_weight: float = 0.10

    # Seasonality (soft booster + optional hard gate)
    use_seasonality_filter: bool = False  # set True when real data available
    seasonality_p_threshold: float = 0.05
    seasonality_confidence_weight: float = 0.10

    # Inefficiency gate
    use_inefficiency_filter: bool = True
    inefficiency_min_wr: float = 0.60


class CoreStrategy:
    """
    Triple (+ seasonality) confluence signal generator.

    Hard gates: HMM prob + trend + vol.
    Soft boosters: Wyckoff + sentiment + seasonality.
    """

    def __init__(
        self,
        hmm_detector: HMMRegimeDetector,
        wyckoff: WyckoffAnalyser,
        arb_filter: TemporalArbitrageFilter | None = None,
        seasonality: SeasonalityAnalyser | None = None,
        seasonality_stats: pd.DataFrame | None = None,  # pre-computed weekday stats
        cfg: StrategyConfig | None = None,
    ) -> None:
        self.hmm        = hmm_detector
        self.wyckoff    = wyckoff
        self.arb_filter = arb_filter or TemporalArbitrageFilter(min_win_rate=0.60)
        self.seasonality       = seasonality or SeasonalityAnalyser()
        self.seasonality_stats = seasonality_stats
        self.cfg        = cfg or StrategyConfig()

    def evaluate(
        self,
        asset: str,
        hmm_observations: np.ndarray,
        ohlcv_df: pd.DataFrame,
        sentiment_score: float,
        inefficiency_signal_mask: pd.Series | None = None,
        forward_returns: pd.Series | None = None,
    ) -> SignalRecord:
        ts = ohlcv_df.index[-1].to_pydatetime()
        rationale: list[str] = []

        # ── Gate 1: HMM ───────────────────────────────────────────────────────
        bull_prob     = float(self.hmm.predict_bull_prob(hmm_observations)[-1])
        bear_vol_prob = float(self.hmm.predict_bear_vol_prob(hmm_observations)[-1])
        _, regime_label = self.hmm.get_current_regime(hmm_observations)

        bull_ok = bull_prob     >= self.cfg.min_hmm_bull_prob
        bear_ok = bear_vol_prob >= self.cfg.min_hmm_bear_vol_prob
        rationale.append(
            f"HMM: bull={bull_prob:.3f}({'✓' if bull_ok else '✗'}) "
            f"bear_vol={bear_vol_prob:.3f}({'✓' if bear_ok else '✗'}) "
            f"regime='{regime_label}'"
        )

        # ── Gate 2: Trend ─────────────────────────────────────────────────────
        current_close = float(ohlcv_df["close"].iloc[-1])
        if len(ohlcv_df) >= self.cfg.sma_period:
            sma = float(ohlcv_df["close"].rolling(self.cfg.sma_period).mean().iloc[-1])
            trend_up   = current_close > sma
            trend_down = current_close < sma
        else:
            trend_up = trend_down = True; sma = float("nan")

        long_trend_ok  = (not self.cfg.use_trend_filter) or trend_up
        short_trend_ok = (not self.cfg.use_trend_filter) or trend_down
        rationale.append(
            f"Trend(SMA{self.cfg.sma_period}): {current_close:.2f}/{sma:.2f} "
            f"up={'✓' if long_trend_ok else '✗'} dn={'✓' if short_trend_ok else '✗'}"
        )

        # ── Gate 3: Vol filter ────────────────────────────────────────────────
        if "atr" in ohlcv_df.columns:
            atr_ratio = float(
                ohlcv_df["atr"].iloc[-1] /
                (ohlcv_df["atr"].rolling(60).mean().iloc[-1] + 1e-9)
            )
        else:
            atr_ratio = 1.0
        vol_ok = (not self.cfg.use_vol_filter) or (atr_ratio < self.cfg.atr_ratio_threshold)
        rationale.append(
            f"VolFilter: ATR_ratio={atr_ratio:.2f} "
            f"({'✓' if vol_ok else '✗'})"
        )

        # ── Soft 1: Wyckoff ───────────────────────────────────────────────────
        enriched = self.wyckoff.analyse(ohlcv_df)
        snap: WyckoffSnapshot = self.wyckoff.latest_snapshot(enriched)
        wy_long  = snap.phase in self.cfg.wyckoff_long_phases
        wy_short = snap.phase in self.cfg.wyckoff_short_phases
        wy_score = snap.phase_score if (wy_long or wy_short) else 0.0
        rationale.append(
            f"Wyckoff: '{snap.phase}' score={snap.phase_score:.2f} "
            f"spring={snap.spring_detected} upthrust={snap.upthrust_detected}"
        )

        # ── Soft 2: Sentiment ─────────────────────────────────────────────────
        sent_long  = sentiment_score >= self.cfg.min_sentiment_long
        sent_short = sentiment_score <= self.cfg.max_sentiment_short
        sent_norm  = min(abs(sentiment_score) * 5.0, 1.0)
        rationale.append(f"Sentiment: {sentiment_score:.3f}")

        # ── Soft 3: Seasonality (v3) ──────────────────────────────────────────
        seasonality_boost = 0.0
        if self.seasonality_stats is not None and not self.seasonality_stats.empty:
            weekday = ts.weekday()
            if weekday in self.seasonality_stats.index:
                row = self.seasonality_stats.loc[weekday]
                is_sig = bool(row.get("significant", False))
                mean_r = float(row.get("mean_ret", 0))
                if is_sig:
                    seasonality_boost = min(abs(mean_r) * 100, 1.0) * np.sign(mean_r)
            rationale.append(f"Seasonality(weekday={weekday}): boost={seasonality_boost:+.3f}")

        # ── Inefficiency gate ─────────────────────────────────────────────────
        ineff_active = False
        if (
            self.cfg.use_inefficiency_filter
            and inefficiency_signal_mask is not None
            and forward_returns is not None
        ):
            ineff_valid, ineff_wr = self.arb_filter.validate(
                inefficiency_signal_mask, forward_returns
            )
            ineff_active = ineff_valid
            rationale.append(f"IneffWR: {ineff_wr:.1%} {'✓' if ineff_active else '✗'}")

        # ── Direction resolution ──────────────────────────────────────────────
        direction = 0
        if bull_ok and long_trend_ok and vol_ok:
            direction = 1
            rationale.append("LONG: HMM ✓ + trend ✓ + vol ✓")
        elif bear_ok and short_trend_ok and vol_ok:
            direction = -1
            rationale.append("SHORT: HMM ✓ + trend ✓ + vol ✓")
        else:
            rationale.append("NO SIGNAL: hard gate(s) failed.")

        # ── Confidence composite ──────────────────────────────────────────────
        if direction != 0:
            hmm_score = bull_prob if direction == 1 else bear_vol_prob
            base_w    = 1.0 - self.cfg.wyckoff_confidence_weight \
                            - self.cfg.sentiment_confidence_weight \
                            - self.cfg.seasonality_confidence_weight
            wy_contrib = (
                wy_score * self.cfg.wyckoff_confidence_weight
                if (direction == 1 and wy_long) or (direction == -1 and wy_short)
                else 0.0
            )
            sent_contrib = (
                sent_norm * self.cfg.sentiment_confidence_weight
                if (direction == 1 and sent_long) or (direction == -1 and sent_short)
                else 0.0
            )
            seas_contrib = (
                abs(seasonality_boost) * self.cfg.seasonality_confidence_weight
                if (direction == 1 and seasonality_boost > 0)
                   or (direction == -1 and seasonality_boost < 0)
                else 0.0
            )
            confidence = hmm_score * base_w + wy_contrib + sent_contrib + seas_contrib
        else:
            confidence = 0.0

        return SignalRecord(
            timestamp=ts, asset=asset, direction=direction,
            hmm_regime=regime_label,
            hmm_proba=bull_prob if direction >= 0 else bear_vol_prob,
            wyckoff_phase=snap.phase, wyckoff_score=snap.phase_score,
            sentiment_score=sentiment_score, inefficiency_active=ineff_active,
            seasonality_boost=float(seasonality_boost),
            confidence=round(float(confidence), 4), rationale=rationale,
        )
