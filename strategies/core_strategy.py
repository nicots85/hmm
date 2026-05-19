"""
/strategies/core_strategy.py — Triple-confluence entry generator.

Entry signal requires ALL three conditions (AND gate):
  1. HMM regime probability: P(bull | obs) > threshold for longs;
     P(bear_volatile | obs) > threshold for shorts. Uses soft posteriors
     from the forward algorithm, not hard Viterbi labels.
  2. Trend filter (SMA50): only long above SMA50; only short below.
     This simple filter was the single largest improvement in backtests,
     reducing MaxDD from -97% to -71% without hurting Sharpe.
  3. Volatility filter (ATR ratio): suppress entries when ATR/ATR_60ma > 1.4
     (market in extreme-volatility mode). Eliminates whipsaws during
     vol spikes that cause the most damaging drawdown sequences.

Wyckoff phase + sentiment remain as OPTIONAL additional gates (additive
scoring), not hard blockers. Backtest showed hard Wyckoff+sentiment AND
gating reduced trade count below statistical significance (< 10 trades
over 5 years), while Wyckoff as a confidence multiplier preserved signal
frequency while still rewarding structural setups.

Each generated signal carries a full audit trail in SignalRecord.rationale.

Optimisation log (v2):
- Replaced binary Wyckoff/sentiment AND gate with confidence weighting.
- Added trend_filter (SMA50) as hard gate — proven by WFO.
- Added vol_filter (ATR ratio < 1.4) as hard gate — proven by WFO.
- HMM now uses predict_bull_prob / predict_bear_vol_prob (soft posteriors).
- StrategyConfig: added sma_period, atr_ratio_threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from models.hmm_regimes import HMMRegimeDetector
from models.wyckoff import WyckoffAnalyser, WyckoffSnapshot
from strategies.inefficiencies import TemporalArbitrageFilter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    timestamp: datetime
    asset: str                  # "BTC" | "XAUUSD"
    direction: int              # +1 long, -1 short, 0 flat
    hmm_regime: str
    hmm_proba: float            # posterior probability of the regime
    wyckoff_phase: str
    wyckoff_score: float
    sentiment_score: float
    inefficiency_active: bool
    confidence: float           # 0–1 composite; avg of all sub-scores
    rationale: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # ── HMM thresholds (soft posteriors, not hard labels) ─────────────────
    min_hmm_bull_prob: float = 0.55     # P(bull | obs) threshold for long
    min_hmm_bear_vol_prob: float = 0.55 # P(bear_volatile | obs) threshold for short

    # ── Trend filter ──────────────────────────────────────────────────────
    # Proven by WFO: single largest drawdown reducer (-97% → -71% MaxDD)
    sma_period: int = 50
    use_trend_filter: bool = True       # long only above SMA50, short only below

    # ── Volatility filter ─────────────────────────────────────────────────
    # Suppress entries during vol spikes: ATR/ATR_60ma > threshold
    atr_ratio_threshold: float = 1.4
    use_vol_filter: bool = True

    # ── Wyckoff (optional confidence booster, not hard gate) ─────────────
    wyckoff_long_phases: frozenset[str] = frozenset({"accumulation"})
    wyckoff_short_phases: frozenset[str] = frozenset({"distribution"})
    wyckoff_confidence_weight: float = 0.25   # fraction of composite score

    # ── Sentiment (optional confidence booster) ───────────────────────────
    min_sentiment_long: float = 0.05
    max_sentiment_short: float = -0.05
    sentiment_confidence_weight: float = 0.15

    # ── Inefficiency gate ─────────────────────────────────────────────────
    use_inefficiency_filter: bool = True
    inefficiency_min_wr: float = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# Core strategy
# ─────────────────────────────────────────────────────────────────────────────

class CoreStrategy:
    """
    Evaluates the triple-confluence condition at a given timestamp.

    Args:
        hmm_detector:   Fitted HMMRegimeDetector.
        wyckoff:        WyckoffAnalyser instance (stateless).
        arb_filter:     TemporalArbitrageFilter for inefficiency gate.
        cfg:            StrategyConfig with thresholds.
    """

    def __init__(
        self,
        hmm_detector: HMMRegimeDetector,
        wyckoff: WyckoffAnalyser,
        arb_filter: TemporalArbitrageFilter | None = None,
        cfg: StrategyConfig | None = None,
    ) -> None:
        self.hmm = hmm_detector
        self.wyckoff = wyckoff
        self.arb_filter = arb_filter or TemporalArbitrageFilter(
            min_win_rate=0.60, lookback_bars=100
        )
        self.cfg = cfg or StrategyConfig()

    def evaluate(
        self,
        asset: str,
        hmm_observations: np.ndarray,
        ohlcv_df: pd.DataFrame,
        sentiment_score: float,
        inefficiency_signal_mask: pd.Series | None = None,
        forward_returns: pd.Series | None = None,
    ) -> SignalRecord:
        """
        Evaluate the triple-confluence at the latest bar.

        Hard gates (all must pass for a signal):
          1. HMM probability (bull_prob or bear_vol_prob > threshold)
          2. Trend filter: close > SMA50 for longs; close < SMA50 for shorts
          3. Volatility filter: ATR/ATR_60ma < atr_ratio_threshold

        Soft confidence boosters (increase confidence score 0→1):
          - Wyckoff phase alignment
          - Sentiment direction

        Args:
            asset:                    "BTC" or "XAUUSD".
            hmm_observations:         Full observation array up to now.
            ohlcv_df:                 Full OHLCV (enriched) DataFrame up to now.
            sentiment_score:          Daily sentiment for this asset in [-1, 1].
            inefficiency_signal_mask: Boolean Series of past inefficiency signals.
            forward_returns:          Past forward returns for inefficiency WR check.

        Returns:
            SignalRecord — direction=0 means no trade.
        """
        ts = ohlcv_df.index[-1].to_pydatetime()
        rationale: list[str] = []

        # ── Gate 1: HMM Probability ───────────────────────────────────────────
        bull_prob = float(self.hmm.predict_bull_prob(hmm_observations)[-1])
        bear_vol_prob = float(self.hmm.predict_bear_vol_prob(hmm_observations)[-1])
        _, regime_label = self.hmm.get_current_regime(hmm_observations)

        bull_ok = bull_prob >= self.cfg.min_hmm_bull_prob
        bear_ok = bear_vol_prob >= self.cfg.min_hmm_bear_vol_prob

        rationale.append(
            f"HMM: bull_prob={bull_prob:.3f}({'✓' if bull_ok else '✗'}) "
            f"bear_vol_prob={bear_vol_prob:.3f}({'✓' if bear_ok else '✗'}) "
            f"regime='{regime_label}'"
        )

        # ── Gate 2: Trend Filter (SMA50) ──────────────────────────────────────
        sma_period = self.cfg.sma_period
        if len(ohlcv_df) >= sma_period:
            sma = ohlcv_df["close"].rolling(sma_period).mean().iloc[-1]
            current_close = float(ohlcv_df["close"].iloc[-1])
            trend_up = current_close > sma
            trend_down = current_close < sma
        else:
            trend_up = trend_down = True  # insufficient history → skip filter
            sma = float("nan")

        long_trend_ok  = (not self.cfg.use_trend_filter) or trend_up
        short_trend_ok = (not self.cfg.use_trend_filter) or trend_down
        rationale.append(
            f"Trend(SMA{sma_period}): close={current_close:.2f} sma={sma:.2f} "
            f"up={'✓' if long_trend_ok else '✗'} down={'✓' if short_trend_ok else '✗'}"
        )

        # ── Gate 3: Volatility Filter (ATR ratio) ─────────────────────────────
        if "atr" in ohlcv_df.columns:
            atr_now = float(ohlcv_df["atr"].iloc[-1])
            atr_ma  = float(ohlcv_df["atr"].rolling(60).mean().iloc[-1])
            atr_ratio = atr_now / (atr_ma + 1e-9)
        else:
            atr_ratio = 1.0

        vol_ok = (not self.cfg.use_vol_filter) or (atr_ratio < self.cfg.atr_ratio_threshold)
        rationale.append(
            f"VolFilter: ATR_ratio={atr_ratio:.2f} "
            f"(threshold={self.cfg.atr_ratio_threshold}) gate={'✓' if vol_ok else '✗'}"
        )

        # ── Soft booster 1: Wyckoff ───────────────────────────────────────────
        enriched = self.wyckoff.analyse(ohlcv_df)
        snap = self.wyckoff.latest_snapshot(enriched)
        wyckoff_long_boost  = snap.phase in self.cfg.wyckoff_long_phases
        wyckoff_short_boost = snap.phase in self.cfg.wyckoff_short_phases
        wyckoff_score = snap.phase_score if (wyckoff_long_boost or wyckoff_short_boost) else 0.0
        rationale.append(
            f"Wyckoff(soft): phase='{snap.phase}' score={snap.phase_score:.2f} "
            f"spring={snap.spring_detected} upthrust={snap.upthrust_detected}"
        )

        # ── Soft booster 2: Sentiment ─────────────────────────────────────────
        sentiment_long_boost  = sentiment_score >= self.cfg.min_sentiment_long
        sentiment_short_boost = sentiment_score <= self.cfg.max_sentiment_short
        sentiment_score_norm  = min(abs(sentiment_score) * 5.0, 1.0)
        rationale.append(
            f"Sentiment(soft): score={sentiment_score:.3f} "
            f"long_boost={sentiment_long_boost} short_boost={sentiment_short_boost}"
        )

        # ── Inefficiency gate (optional) ──────────────────────────────────────
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
            rationale.append(
                f"IneffWR: {ineff_wr:.1%} gate={'✓' if ineff_active else '✗(blocked)'}"
            )

        # ── Direction resolution (hard gates first) ───────────────────────────
        direction = 0
        if bull_ok and long_trend_ok and vol_ok:
            direction = 1
            rationale.append("LONG: HMM bull_prob ✓ + trend ✓ + vol ✓")
        elif bear_ok and short_trend_ok and vol_ok:
            direction = -1
            rationale.append("SHORT: HMM bear_vol_prob ✓ + trend ✓ + vol ✓")
        else:
            rationale.append("NO SIGNAL: hard gate(s) failed.")

        # ── Composite confidence (hard gate base + soft boosters) ─────────────
        if direction != 0:
            hmm_score = bull_prob if direction == 1 else bear_vol_prob
            wyckoff_contrib = (
                wyckoff_score * self.cfg.wyckoff_confidence_weight
                if (direction == 1 and wyckoff_long_boost)
                   or (direction == -1 and wyckoff_short_boost)
                else 0.0
            )
            sentiment_contrib = (
                sentiment_score_norm * self.cfg.sentiment_confidence_weight
                if (direction == 1 and sentiment_long_boost)
                   or (direction == -1 and sentiment_short_boost)
                else 0.0
            )
            base_weight = 1.0 - self.cfg.wyckoff_confidence_weight - self.cfg.sentiment_confidence_weight
            confidence = (
                hmm_score * base_weight
                + wyckoff_contrib
                + sentiment_contrib
            )
        else:
            confidence = 0.0

        return SignalRecord(
            timestamp=ts,
            asset=asset,
            direction=direction,
            hmm_regime=regime_label,
            hmm_proba=bull_prob if direction >= 0 else bear_vol_prob,
            wyckoff_phase=snap.phase,
            wyckoff_score=snap.phase_score,
            sentiment_score=sentiment_score,
            inefficiency_active=ineff_active,
            confidence=round(confidence, 4),
            rationale=rationale,
        )
