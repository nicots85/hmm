"""
/strategies/core_strategy.py — Triple-confluence entry generator.

Entry signal requires ALL three conditions (AND gate):
  1. HMM regime is favourable (bull_calm or bull_volatile for longs;
     bear_calm or bear_volatile for shorts).
  2. Wyckoff phase confirms: accumulation + spring for longs;
     distribution + upthrust for shorts.
  3. Geopolitical / sentiment filter: daily sentiment score > 0 (long)
     or < 0 (short) for the relevant asset.

The design intentionally makes entries rare and high-conviction.
False-positive risk is minimised at the cost of reduced trade frequency.

Soft thresholds (e.g. HMM posterior probability, Wyckoff phase_score)
are configurable so the strategy can be tuned during WFO without changing
the structural logic.

Each generated signal carries a full audit trail (why it was triggered)
stored in SignalRecord.rationale for post-trade analysis.
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
    # HMM thresholds
    bull_regimes: frozenset[str] = frozenset({"bull_calm", "bull_volatile"})
    bear_regimes: frozenset[str] = frozenset({"bear_calm", "bear_volatile"})
    min_hmm_proba: float = 0.55     # minimum posterior for regime confidence

    # Wyckoff thresholds
    long_phases: frozenset[str] = frozenset({"accumulation"})
    short_phases: frozenset[str] = frozenset({"distribution"})
    min_wyckoff_score: float = 0.65

    # Sentiment
    min_sentiment_long: float = 0.05     # weak positive sentiment accepted
    max_sentiment_short: float = -0.05   # weak negative sentiment accepted

    # Inefficiency gate
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

        # ── Gate 1: HMM Regime ────────────────────────────────────────────────
        regime_idx, regime_label = self.hmm.get_current_regime(hmm_observations)
        posteriors = self.hmm.predict_proba(hmm_observations)
        hmm_proba = float(posteriors[-1, regime_idx])

        is_bull_regime = regime_label in self.cfg.bull_regimes
        is_bear_regime = regime_label in self.cfg.bear_regimes
        hmm_ok = (is_bull_regime or is_bear_regime) and hmm_proba >= self.cfg.min_hmm_proba

        rationale.append(
            f"HMM: regime='{regime_label}', proba={hmm_proba:.2f}, "
            f"gate={'PASS' if hmm_ok else 'FAIL'}"
        )

        # ── Gate 2: Wyckoff Phase ─────────────────────────────────────────────
        enriched = self.wyckoff.analyse(ohlcv_df)
        snap: WyckoffSnapshot = self.wyckoff.latest_snapshot(enriched)

        is_acc = snap.phase in self.cfg.long_phases
        is_dist = snap.phase in self.cfg.short_phases
        wyckoff_long_ok = is_acc and snap.phase_score >= self.cfg.min_wyckoff_score
        wyckoff_short_ok = is_dist and snap.phase_score >= self.cfg.min_wyckoff_score

        wyckoff_ok = wyckoff_long_ok or wyckoff_short_ok
        rationale.append(
            f"Wyckoff: phase='{snap.phase}', score={snap.phase_score:.2f}, "
            f"spring={snap.spring_detected}, upthrust={snap.upthrust_detected}, "
            f"gate={'PASS' if wyckoff_ok else 'FAIL'}"
        )

        # ── Gate 3: Sentiment ─────────────────────────────────────────────────
        sentiment_long_ok = sentiment_score >= self.cfg.min_sentiment_long
        sentiment_short_ok = sentiment_score <= self.cfg.max_sentiment_short
        sentiment_ok = sentiment_long_ok or sentiment_short_ok

        rationale.append(
            f"Sentiment: score={sentiment_score:.3f}, "
            f"long_ok={sentiment_long_ok}, short_ok={sentiment_short_ok}, "
            f"gate={'PASS' if sentiment_ok else 'FAIL'}"
        )

        # ── Gate 4 (optional): Inefficiency walk-forward WR ──────────────────
        ineff_active = False
        if self.cfg.use_inefficiency_filter and (
            inefficiency_signal_mask is not None and forward_returns is not None
        ):
            ineff_valid, ineff_wr = self.arb_filter.validate(
                inefficiency_signal_mask, forward_returns
            )
            ineff_active = ineff_valid
            rationale.append(
                f"Inefficiency WR: {ineff_wr:.1%}, "
                f"gate={'PASS' if ineff_active else 'FAIL (blocked)'}"
            )

        # ── Direction resolution ──────────────────────────────────────────────
        direction = 0
        if hmm_ok and wyckoff_ok and sentiment_ok:
            if is_bull_regime and wyckoff_long_ok and sentiment_long_ok:
                direction = 1
                rationale.append("LONG signal confirmed by triple confluence.")
            elif is_bear_regime and wyckoff_short_ok and sentiment_short_ok:
                direction = -1
                rationale.append("SHORT signal confirmed by triple confluence.")
        else:
            rationale.append("No signal: one or more gates failed.")

        # ── Composite confidence ──────────────────────────────────────────────
        sub_scores = [hmm_proba, snap.phase_score, min(abs(sentiment_score) * 5, 1.0)]
        confidence = float(np.mean(sub_scores)) if direction != 0 else 0.0

        return SignalRecord(
            timestamp=ts,
            asset=asset,
            direction=direction,
            hmm_regime=regime_label,
            hmm_proba=hmm_proba,
            wyckoff_phase=snap.phase,
            wyckoff_score=snap.phase_score,
            sentiment_score=sentiment_score,
            inefficiency_active=ineff_active,
            confidence=round(confidence, 4),
            rationale=rationale,
        )
