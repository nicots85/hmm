"""
/execution/risk_manager.py — Dynamic risk management engine.

Three components:

1. KellyPositionSizer
   - Fractional Kelly criterion for position sizing.
   - Kelly fraction f* = (p·b - q) / b, where b = avg_win/avg_loss,
     p = win_rate, q = 1-p.
   - We apply a "fractional" multiplier (default 0.25) to the raw Kelly
     output because the raw Kelly is maximally aggressive and causes
     intolerable drawdowns in practice.
   - Hard cap: position size never exceeds max_position_risk_pct of NAV.

2. DynamicStopLoss
   PHILOSOPHY (INQUEBRANTABLE): Stop loss placement is always derived from
   market structure and volatility. There is NO breakeven mechanic, NO
   "move-to-entry" logic. Every SL adjustment must be justified by one of:
     a) ATR expansion (volatility widening requires wider SL).
     b) A new structural level (swing low/high migrating).
     c) Regime change detected by the HMM (risk re-assessment).
   The SL is calculated as: entry ± (atr_multiplier × current_ATR),
   constrained to not move AGAINST the trade (trailing-only semantics).

3. HedgeManager
   - Monitors rolling BTC/XAUUSD return correlation.
   - If |correlation| > threshold and regime is "bear_volatile",
     opens a partial XAUUSD long hedge against a BTC short (or vice versa).
   - Hedge size = position_size × |correlation| × hedge_ratio.
   - Correlation is computed on log-returns to avoid spurious level effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Kelly Position Sizer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KellyResult:
    raw_kelly: float          # uncapped Kelly fraction
    fractional_kelly: float   # after applying kelly_fraction multiplier
    position_size_pct: float  # final % of NAV to risk (capped)
    position_size_usd: float  # absolute dollar size given NAV


class KellyPositionSizer:
    """
    Fractional Kelly criterion for position sizing.

    Usage:
        sizer = KellyPositionSizer(nav=50_000)
        result = sizer.compute(win_rate=0.55, avg_win=0.03, avg_loss=0.015)
        print(result.position_size_usd)
    """

    def __init__(
        self,
        nav: float,
        kelly_fraction: float | None = None,
        max_risk_pct: float | None = None,
    ) -> None:
        self.nav = nav
        self.kelly_fraction = kelly_fraction or settings.kelly_fraction
        self.max_risk_pct = max_risk_pct or settings.max_position_risk_pct

    def compute(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> KellyResult:
        """
        Compute fractional Kelly position size.

        Args:
            win_rate: historical P(win), in (0, 1).
            avg_win:  mean return on winning trades (positive scalar, e.g. 0.03 = 3%).
            avg_loss: mean return on losing trades (positive scalar, e.g. 0.015 = 1.5%).

        Returns:
            KellyResult with all sizing metrics.

        Raises:
            ValueError if inputs are degenerate (e.g. avg_loss == 0).
        """
        if avg_loss <= 0:
            raise ValueError("avg_loss must be > 0.")
        if not 0 < win_rate < 1:
            raise ValueError("win_rate must be in (0, 1).")

        b = avg_win / avg_loss   # reward-to-risk ratio
        p = win_rate
        q = 1.0 - p

        raw_kelly = (p * b - q) / b   # Kelly formula
        # Negative Kelly → no edge; don't trade
        raw_kelly = max(raw_kelly, 0.0)

        fractional = raw_kelly * self.kelly_fraction
        capped = min(fractional, self.max_risk_pct)

        return KellyResult(
            raw_kelly=round(raw_kelly, 6),
            fractional_kelly=round(fractional, 6),
            position_size_pct=round(capped, 6),
            position_size_usd=round(capped * self.nav, 2),
        )

    def update_nav(self, new_nav: float) -> None:
        """Update NAV for mark-to-market recalculation."""
        self.nav = new_nav


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dynamic Stop Loss  (structure- and volatility-anchored ONLY)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StopLossLevel:
    stop_price: float
    distance_atr: float          # SL distance expressed in ATR units
    anchor_type: str             # "atr_multiplier" | "structural" | "regime_change"
    rationale: str               # human-readable justification


class DynamicStopLoss:
    """
    Compute and update stop losses anchored strictly to market context.

    RULE: The stop loss price can only move in the direction that REDUCES
    risk (i.e., trailing stop only).  It may NEVER be moved back to entry
    price as a default or arbitrary "safety" mechanic.

    SL placement logic (precedence order):
    1. Structural: if a significant swing low/high is within 1.5×ATR
       of the entry, use that level (minus/plus a small buffer).
    2. ATR-based: entry ± (atr_multiplier × ATR) as the baseline.
    3. Regime-adjusted: if HMM detects a bear_volatile regime, widen
       the ATR multiplier by 1.5× to account for increased noise.
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        structural_buffer_pct: float = 0.002,  # 0.2% beyond structural level
    ) -> None:
        self.atr_multiplier = atr_multiplier
        self.structural_buffer = structural_buffer_pct

    def initial_stop(
        self,
        entry_price: float,
        direction: int,                    # +1 long, -1 short
        current_atr: float,
        swing_low: float | None = None,
        swing_high: float | None = None,
        regime_label: str = "",
    ) -> StopLossLevel:
        """
        Calculate the initial stop loss at trade entry.

        Args:
            entry_price:   Execution price.
            direction:     +1 for long, -1 for short.
            current_atr:   ATR value at the time of entry.
            swing_low:     Nearest swing low (for long trades).
            swing_high:    Nearest swing high (for short trades).
            regime_label:  HMM regime label; "bear_volatile" widens SL.

        Returns:
            StopLossLevel with full rationale.
        """
        # Regime-adjusted ATR multiplier
        multiplier = self.atr_multiplier
        if "volatile" in regime_label.lower():
            multiplier *= 1.5
            rationale_prefix = f"Regime '{regime_label}': ATR multiplier widened to {multiplier:.1f}x. "
        else:
            rationale_prefix = f"Regime '{regime_label}': standard ATR multiplier {multiplier:.1f}x. "

        # ATR-based baseline
        atr_stop = entry_price - direction * multiplier * current_atr

        # Structural override if swing level is available and relevant
        if direction > 0 and swing_low is not None:
            structural_stop = swing_low * (1 - self.structural_buffer)
            if abs(entry_price - swing_low) <= 1.5 * current_atr:
                stop_price = structural_stop
                return StopLossLevel(
                    stop_price=round(stop_price, 8),
                    distance_atr=abs(entry_price - stop_price) / (current_atr + 1e-12),
                    anchor_type="structural",
                    rationale=rationale_prefix + f"Structural SL below swing low {swing_low:.2f} "
                              f"with {self.structural_buffer*100:.1f}% buffer.",
                )

        if direction < 0 and swing_high is not None:
            structural_stop = swing_high * (1 + self.structural_buffer)
            if abs(swing_high - entry_price) <= 1.5 * current_atr:
                stop_price = structural_stop
                return StopLossLevel(
                    stop_price=round(stop_price, 8),
                    distance_atr=abs(stop_price - entry_price) / (current_atr + 1e-12),
                    anchor_type="structural",
                    rationale=rationale_prefix + f"Structural SL above swing high {swing_high:.2f} "
                              f"with {self.structural_buffer*100:.1f}% buffer.",
                )

        # Fallback: ATR-based
        return StopLossLevel(
            stop_price=round(atr_stop, 8),
            distance_atr=multiplier,
            anchor_type="atr_multiplier",
            rationale=rationale_prefix + f"ATR-based SL: entry {entry_price:.2f} "
                      f"± {multiplier:.1f} × ATR({current_atr:.4f}).",
        )

    def trail_stop(
        self,
        current_stop: StopLossLevel,
        direction: int,
        current_price: float,
        current_atr: float,
        new_swing_low: float | None = None,
        new_swing_high: float | None = None,
        regime_label: str = "",
    ) -> StopLossLevel:
        """
        Update the stop loss only if the new level reduces risk vs. the
        existing stop. Trailing is strictly one-directional.

        This method will NEVER return a stop price further from the current
        price than the existing stop (i.e., it never INCREASES risk exposure).
        """
        new_sl = self.initial_stop(
            entry_price=current_price,
            direction=direction,
            current_atr=current_atr,
            swing_low=new_swing_low,
            swing_high=new_swing_high,
            regime_label=regime_label,
        )

        # Enforce trailing-only semantics
        if direction > 0:
            # For longs: new stop must be HIGHER than existing (never lower)
            if new_sl.stop_price <= current_stop.stop_price:
                logger.debug(
                    "Trail rejected: new SL %.4f ≤ current SL %.4f (long).",
                    new_sl.stop_price, current_stop.stop_price
                )
                return current_stop
        else:
            # For shorts: new stop must be LOWER than existing (never higher)
            if new_sl.stop_price >= current_stop.stop_price:
                logger.debug(
                    "Trail rejected: new SL %.4f ≥ current SL %.4f (short).",
                    new_sl.stop_price, current_stop.stop_price
                )
                return current_stop

        logger.info(
            "SL trailed from %.4f → %.4f (direction=%+d). Reason: %s",
            current_stop.stop_price, new_sl.stop_price, direction, new_sl.rationale
        )
        return new_sl


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hedge Manager
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HedgeSignal:
    hedge_active: bool
    hedge_asset: str              # "XAUUSD" or "BTC"
    hedge_direction: int          # +1 long, -1 short
    hedge_size_pct: float         # as fraction of primary position
    correlation: float
    rationale: str


class HedgeManager:
    """
    Monitors BTC/Gold rolling correlation and recommends hedge positions.

    Hedge logic:
    - If correlation > +threshold and primary is long BTC: hedge with short GOLD
      (they co-move; GOLD short partially offsets a BTC drawdown in a risk-off event).
    - If correlation < -threshold: assets diverge; no structural hedge.
    - Hedge size is proportional to |correlation| × hedge_ratio.
    - Hedge is only opened in "bear_volatile" or "bear_calm" HMM regimes.
    """

    def __init__(
        self,
        correlation_window: int | None = None,
        correlation_threshold: float = 0.5,
        hedge_ratio: float = 0.3,
    ) -> None:
        self.window = correlation_window or settings.hedge_correlation_window
        self.threshold = correlation_threshold
        self.hedge_ratio = hedge_ratio

    def evaluate(
        self,
        btc_returns: pd.Series,
        gold_returns: pd.Series,
        primary_direction: int,
        primary_asset: str,
        current_regime: str,
    ) -> HedgeSignal:
        """
        Compute hedge recommendation given current market state.

        Args:
            btc_returns:       Series of BTC log-returns.
            gold_returns:      Series of XAUUSD log-returns.
            primary_direction: Direction of the primary trade (+1 / -1).
            primary_asset:     "BTC" or "XAUUSD".
            current_regime:    HMM regime label.

        Returns:
            HedgeSignal with hedge parameters or hedge_active=False.
        """
        # Align and compute rolling correlation
        aligned = pd.concat([btc_returns, gold_returns], axis=1).dropna()
        aligned.columns = ["btc", "gold"]

        if len(aligned) < self.window:
            return HedgeSignal(
                hedge_active=False, hedge_asset="", hedge_direction=0,
                hedge_size_pct=0.0, correlation=0.0,
                rationale="Insufficient history for correlation."
            )

        corr = float(aligned["btc"].iloc[-self.window:].corr(aligned["gold"].iloc[-self.window:]))

        # Only hedge in bearish regimes
        is_bear_regime = "bear" in current_regime.lower()
        if not is_bear_regime:
            return HedgeSignal(
                hedge_active=False, hedge_asset="", hedge_direction=0,
                hedge_size_pct=0.0, correlation=corr,
                rationale=f"Regime '{current_regime}' is not bearish — no hedge required."
            )

        if abs(corr) < self.threshold:
            return HedgeSignal(
                hedge_active=False, hedge_asset="", hedge_direction=0,
                hedge_size_pct=0.0, correlation=corr,
                rationale=f"Correlation {corr:.2f} below threshold {self.threshold:.2f}."
            )

        # Determine hedge instrument and direction
        if primary_asset == "BTC":
            hedge_asset = "XAUUSD"
            # Positive corr: BTC and Gold move together.
            # Short Gold offsets BTC long loss when both fall.
            hedge_direction = -primary_direction if corr > 0 else primary_direction
        else:
            hedge_asset = "BTC"
            hedge_direction = -primary_direction if corr > 0 else primary_direction

        hedge_size = abs(corr) * self.hedge_ratio

        return HedgeSignal(
            hedge_active=True,
            hedge_asset=hedge_asset,
            hedge_direction=hedge_direction,
            hedge_size_pct=round(hedge_size, 4),
            correlation=round(corr, 4),
            rationale=(
                f"Rolling {self.window}-bar correlation={corr:.2f}. "
                f"Regime='{current_regime}'. Hedge {hedge_size*100:.1f}% "
                f"position in {hedge_asset} direction={hedge_direction:+d}."
            ),
        )
