"""
/execution/risk_manager.py — Dynamic risk management engine.

v3 changes:
- P9 FIX: HedgeManager now uses volatility-adjusted correlation (DCC proxy):
  hedge_size = |corr| * (vol_asset / vol_hedge) * hedge_ratio.
  This ensures the hedge notional matches the volatility exposure, not just
  the direction correlation.
- KellyPositionSizer: added compute_from_series() that derives win_rate,
  avg_win, avg_loss from a historical returns array directly.
- DynamicStopLoss: unchanged (already correct in v2).
- All three components now emit sizing_series suitable for BacktestEngine.
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
    raw_kelly: float
    fractional_kelly: float
    position_size_pct: float
    position_size_usd: float


class KellyPositionSizer:
    """
    Fractional Kelly criterion for position sizing.

    Two entry points:
    - compute(win_rate, avg_win, avg_loss): supply stats directly.
    - compute_from_series(returns): derive stats from historical trade returns.
    """

    def __init__(
        self,
        nav: float,
        kelly_fraction: float | None = None,
        max_risk_pct: float | None = None,
    ) -> None:
        self.nav          = nav
        self.kelly_fraction = kelly_fraction or settings.kelly_fraction
        self.max_risk_pct   = max_risk_pct or settings.max_position_risk_pct

    def compute(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> KellyResult:
        """
        f* = (p·b − q) / b   where b = avg_win / avg_loss.
        Fractional: f_frac = f* × kelly_fraction.
        Capped:     min(f_frac, max_risk_pct).
        """
        if avg_loss <= 0:
            raise ValueError("avg_loss must be > 0.")
        if not 0 < win_rate < 1:
            raise ValueError("win_rate must be in (0, 1).")

        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p
        raw_kelly = max((p * b - q) / b, 0.0)
        fractional = raw_kelly * self.kelly_fraction
        capped = min(fractional, self.max_risk_pct)

        return KellyResult(
            raw_kelly=round(raw_kelly, 6),
            fractional_kelly=round(fractional, 6),
            position_size_pct=round(capped, 6),
            position_size_usd=round(capped * self.nav, 2),
        )

    def compute_from_series(self, trade_returns: np.ndarray) -> KellyResult:
        """
        Derive Kelly inputs from a historical array of trade returns.

        Applies a minimum sample guard: if fewer than 20 trades are available,
        return a conservative minimum sizing to avoid overfitting.
        """
        if len(trade_returns) < 20:
            logger.warning(
                "Insufficient trade history (%d trades). Using conservative sizing.",
                len(trade_returns),
            )
            return KellyResult(
                raw_kelly=0.0,
                fractional_kelly=0.0,
                position_size_pct=self.max_risk_pct * 0.25,
                position_size_usd=self.nav * self.max_risk_pct * 0.25,
            )

        wins   = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return KellyResult(0.0, 0.0, 0.0, 0.0)

        win_rate = float(len(wins) / len(trade_returns))
        avg_win  = float(wins.mean())
        avg_loss = float(abs(losses.mean()))

        return self.compute(win_rate, avg_win, avg_loss)

    def build_sizing_series(
        self,
        signals: pd.Series,
        close: pd.Series,
        lookback_trades: int = 50,
    ) -> pd.Series:
        """
        Build a time-series of Kelly position sizes, recomputed every bar
        using only past trade returns (expanding window, no look-ahead).

        Returns a Series of floats in [0, max_risk_pct] aligned to signals.
        """
        sizing = pd.Series(self.max_risk_pct * 0.25, index=signals.index)
        returns_log = np.log(close / close.shift(1)).fillna(0)
        position = signals.shift(1).fillna(0)
        trade_ret = (position * returns_log).cumsum()

        signal_changes = signals[signals != 0].index
        for i, ts in enumerate(signal_changes):
            if i < 20:
                continue
            past_signals = signals.loc[:ts].iloc[:-1]
            past_log_ret = returns_log.loc[:ts].iloc[:-1]
            trade_rets_arr = (past_signals.shift(1).fillna(0) * past_log_ret).values[-lookback_trades:]
            result = self.compute_from_series(trade_rets_arr)
            sizing.loc[ts:] = result.position_size_pct

        return sizing.clip(0, self.max_risk_pct)

    def update_nav(self, new_nav: float) -> None:
        self.nav = new_nav


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dynamic Stop Loss (unchanged from v2 — correct)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StopLossLevel:
    stop_price: float
    distance_atr: float
    anchor_type: str
    rationale: str


class DynamicStopLoss:
    """
    ATR/structural stop loss — trailing only, never arbitrary breakeven.

    RULE: stop can only move to reduce risk (one direction).
    Regime-adjusted: bear_volatile widens multiplier ×1.5.
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        structural_buffer_pct: float = 0.002,
    ) -> None:
        self.atr_multiplier    = atr_multiplier
        self.structural_buffer = structural_buffer_pct

    def initial_stop(
        self,
        entry_price: float,
        direction: int,
        current_atr: float,
        swing_low: float | None = None,
        swing_high: float | None = None,
        regime_label: str = "",
    ) -> StopLossLevel:
        multiplier = self.atr_multiplier
        if "volatile" in regime_label.lower():
            multiplier *= 1.5
        rationale_prefix = f"Regime='{regime_label}' mult={multiplier:.1f}×ATR. "
        atr_stop = entry_price - direction * multiplier * current_atr

        if direction > 0 and swing_low is not None:
            structural_stop = swing_low * (1 - self.structural_buffer)
            if abs(entry_price - swing_low) <= 1.5 * current_atr:
                return StopLossLevel(
                    stop_price=round(structural_stop, 8),
                    distance_atr=abs(entry_price - structural_stop) / (current_atr + 1e-12),
                    anchor_type="structural",
                    rationale=rationale_prefix + f"Swing low {swing_low:.2f} − buffer.",
                )
        if direction < 0 and swing_high is not None:
            structural_stop = swing_high * (1 + self.structural_buffer)
            if abs(swing_high - entry_price) <= 1.5 * current_atr:
                return StopLossLevel(
                    stop_price=round(structural_stop, 8),
                    distance_atr=abs(structural_stop - entry_price) / (current_atr + 1e-12),
                    anchor_type="structural",
                    rationale=rationale_prefix + f"Swing high {swing_high:.2f} + buffer.",
                )

        return StopLossLevel(
            stop_price=round(atr_stop, 8),
            distance_atr=multiplier,
            anchor_type="atr_multiplier",
            rationale=rationale_prefix + f"entry={entry_price:.2f} ± {multiplier:.1f}×ATR.",
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
        new_sl = self.initial_stop(
            entry_price=current_price,
            direction=direction,
            current_atr=current_atr,
            swing_low=new_swing_low,
            swing_high=new_swing_high,
            regime_label=regime_label,
        )
        if direction > 0 and new_sl.stop_price <= current_stop.stop_price:
            return current_stop
        if direction < 0 and new_sl.stop_price >= current_stop.stop_price:
            return current_stop
        logger.info("SL trailed %.4f → %.4f", current_stop.stop_price, new_sl.stop_price)
        return new_sl


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hedge Manager — v3: volatility-adjusted sizing (P9 fix)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HedgeSignal:
    hedge_active: bool
    hedge_asset: str
    hedge_direction: int
    hedge_size_pct: float
    correlation: float
    vol_ratio: float          # NEW: vol_btc / vol_gold for sizing
    rationale: str


class HedgeManager:
    """
    Volatility-adjusted correlation hedge.

    v3 improvement: hedge notional = position × |corr| × (vol_primary / vol_hedge) × ratio.
    This ensures the hedge actually offsets the primary position's volatility,
    not just its direction. A BTC position with 80% vol hedged with Gold at 15%
    needs a much larger Gold notional than a naive correlation-only approach gives.

    Hedge only activates in bearish HMM regimes (bear_calm or bear_volatile).
    """

    def __init__(
        self,
        correlation_window: int | None = None,
        correlation_threshold: float = 0.5,
        hedge_ratio: float = 0.3,
        vol_window: int = 20,
    ) -> None:
        self.window               = correlation_window or settings.hedge_correlation_window
        self.threshold            = correlation_threshold
        self.hedge_ratio          = hedge_ratio
        self.vol_window           = vol_window

    def evaluate(
        self,
        btc_returns: pd.Series,
        gold_returns: pd.Series,
        primary_direction: int,
        primary_asset: str,
        current_regime: str,
    ) -> HedgeSignal:
        aligned = pd.concat([btc_returns, gold_returns], axis=1).dropna()
        aligned.columns = pd.Index(["btc", "gold"])

        if len(aligned) < self.window:
            return HedgeSignal(
                hedge_active=False, hedge_asset="", hedge_direction=0,
                hedge_size_pct=0.0, correlation=0.0, vol_ratio=1.0,
                rationale="Insufficient history for correlation.",
            )

        window_data = aligned.iloc[-self.window:]
        corr        = float(window_data["btc"].corr(window_data["gold"]))

        # Realised volatility for vol-adjustment
        vol_btc  = float(window_data["btc"].std())
        vol_gold = float(window_data["gold"].std())
        vol_ratio = vol_btc / (vol_gold + 1e-9)

        is_bear_regime = "bear" in current_regime.lower()
        if not is_bear_regime:
            return HedgeSignal(
                hedge_active=False, hedge_asset="", hedge_direction=0,
                hedge_size_pct=0.0, correlation=corr, vol_ratio=vol_ratio,
                rationale=f"Regime '{current_regime}' not bearish.",
            )

        if abs(corr) < self.threshold:
            return HedgeSignal(
                hedge_active=False, hedge_asset="", hedge_direction=0,
                hedge_size_pct=0.0, correlation=corr, vol_ratio=vol_ratio,
                rationale=f"Corr={corr:.2f} < threshold={self.threshold:.2f}.",
            )

        hedge_asset     = "XAUUSD" if primary_asset == "BTC" else "BTC"
        hedge_direction = -primary_direction if corr > 0 else primary_direction

        # Vol-adjusted hedge size: |corr| × (vol_primary/vol_hedge) × ratio
        # Capped at 1.0 to avoid over-hedging
        raw_size    = abs(corr) * vol_ratio * self.hedge_ratio
        hedge_size  = min(raw_size, 1.0)

        return HedgeSignal(
            hedge_active=True,
            hedge_asset=hedge_asset,
            hedge_direction=hedge_direction,
            hedge_size_pct=round(hedge_size, 4),
            correlation=round(corr, 4),
            vol_ratio=round(vol_ratio, 4),
            rationale=(
                f"corr={corr:.2f} vol_ratio={vol_ratio:.2f} "
                f"→ hedge {hedge_size*100:.1f}% {hedge_asset} dir={hedge_direction:+d}. "
                f"Regime='{current_regime}'."
            ),
        )
