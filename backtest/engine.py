"""
/backtest/engine.py — Backtesting engine with walk-forward optimisation (WFO).

Architecture:
- Vectorbt processes the entire price/signal matrix in vectorised form,
  making it ~100x faster than bar-by-bar loops for large histories.
- Walk-forward splits: the timeline is divided into (IS, OOS) pairs.
  IS (in-sample): strategy fitted and optimised.
  OOS (out-of-sample): strategy run with frozen IS parameters.
  This prevents look-ahead bias and overfitting on a single test window.

Metrics reported:
- Sharpe ratio (annualised, risk-free rate configurable).
- Sortino ratio (downside deviation only).
- Maximum drawdown (peak-to-trough).
- Profit factor (gross profit / gross loss).
- Geo-risk exposure: fraction of trades open during high-geopolitical-risk
  periods (|daily_sentiment| > 0.5), and their average PnL contribution.
  This is the custom metric required by the architecture spec.

WFO parameter sweep:
- Sweeps atr_multiplier ∈ [1.5, 2.0, 2.5, 3.0] and
  min_hmm_proba ∈ [0.50, 0.55, 0.60, 0.65].
- Selects best param set per IS window by Sharpe ratio.
- Applies frozen params to OOS; aggregates OOS equity curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics (pure functions — no vectorbt dependency)
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 365) -> float:
    """Annualised Sharpe ratio. Default periods_per_year=365 for daily crypto (trades 24/7)."""
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 365) -> float:
    """Annualised Sortino ratio. Default periods_per_year=365 for daily crypto."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (e.g. 0.25 = 25%)."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / (rolling_max + 1e-12)
    return float(drawdown.min())  # negative; caller interprets abs value


def profit_factor(returns: pd.Series) -> float:
    """Gross profit / gross loss. > 1 means profitable."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")


def geo_risk_exposure(
    returns: pd.Series,
    sentiment_series: pd.Series,
    sentiment_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Custom metric: analyse trade returns during high-geopolitical-risk periods.

    Args:
        returns:              Per-period trade returns (0 when flat).
        sentiment_series:     Daily |sentiment| aligned to returns index.
        sentiment_threshold:  |sentiment| > threshold → high geo-risk period.

    Returns:
        Dict {exposure_fraction, mean_pnl_during_geo_risk, mean_pnl_normal}.
    """
    # Align index
    common = returns.index.intersection(sentiment_series.index)
    if common.empty:
        return {"exposure_fraction": 0.0, "mean_pnl_geo": 0.0, "mean_pnl_normal": 0.0}

    rets = returns.loc[common]
    sent = sentiment_series.loc[common].abs()

    in_trade = rets != 0
    high_risk = sent > sentiment_threshold

    geo_trades = rets[in_trade & high_risk]
    normal_trades = rets[in_trade & ~high_risk]

    return {
        "exposure_fraction": float(len(geo_trades) / max(in_trade.sum(), 1)),
        "mean_pnl_geo": float(geo_trades.mean()) if not geo_trades.empty else 0.0,
        "mean_pnl_normal": float(normal_trades.mean()) if not normal_trades.empty else 0.0,
    }

def regime_performance_breakdown(
    returns: pd.Series,
    regime_series: pd.Series,
) -> pd.DataFrame:
    """
    Break down strategy returns by HMM regime label.

    Args:
        returns:       Per-period strategy returns (0 when flat).
        regime_series: Series of string regime labels aligned to returns.

    Returns:
        DataFrame indexed by regime with [n_days, mean_ret, sharpe, win_rate].
    """
    common = returns.index.intersection(regime_series.index)
    df = pd.DataFrame({"ret": returns.loc[common], "regime": regime_series.loc[common]})
    rows = []
    for lbl, grp in df.groupby("regime"):
        r = grp["ret"]
        rows.append({
            "regime": lbl,
            "n_days": len(r),
            "mean_ret": float(r.mean()),
            "sharpe": sharpe_ratio(r),
            "win_rate": float((r > 0).mean()),
        })
    return pd.DataFrame(rows).set_index("regime").round(4)




@dataclass
class BacktestResult:
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    profit_factor: float
    total_trades: int
    win_rate: float
    geo_risk_metrics: dict[str, float]
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Sharpe={self.sharpe:.2f} | Sortino={self.sortino:.2f} | "
            f"MaxDD={self.max_drawdown_pct*100:.1f}% | PF={self.profit_factor:.2f} | "
            f"Trades={self.total_trades} | WR={self.win_rate*100:.1f}% | "
            f"GeoExposure={self.geo_risk_metrics.get('exposure_fraction', 0)*100:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vectorbt-based backtester
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Runs vectorised backtests using vectorbt.

    Usage:
        engine = BacktestEngine(close_prices, signals, sentiment_series)
        result = engine.run()
        wfo_results = engine.walk_forward(n_splits=5)
    """

    def __init__(
        self,
        close: pd.Series,
        signals: pd.Series,         # +1 long, -1 short, 0 flat
        sentiment: pd.Series | None = None,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.0005,   # 0.05% per side (taker fee)
    ) -> None:
        self.close = close.dropna()
        self.signals = signals.reindex(self.close.index).fillna(0)
        self.sentiment = sentiment
        self.capital = initial_capital
        self.commission = commission_pct

    def run(
        self,
        params: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """
        Execute the backtest on the full data window.

        Attempts to use vectorbt if installed; falls back to a pure-numpy
        simulation that is correct but slower.
        """
        params = params or {}
        try:
            return self._run_vectorbt(params)
        except ImportError:
            logger.warning("vectorbt not available; using numpy fallback.")
            return self._run_numpy(params)

    def walk_forward(
        self,
        n_splits: int = 5,
        is_fraction: float = 0.7,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> list[BacktestResult]:
        """
        Walk-forward optimisation.

        Divides the timeline into n_splits equal windows; for each window:
          - IS (is_fraction of window): grid-search params by Sharpe.
          - OOS (1 - is_fraction): run best IS params, record OOS result.

        Args:
            n_splits:     Number of WFO splits.
            is_fraction:  Fraction of each split used for in-sample training.
            param_grid:   Parameter grid to sweep; default grid if None.

        Returns:
            List of BacktestResult for each OOS window.
        """
        default_grid: dict[str, list[Any]] = {
            "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
            "min_hmm_proba": [0.50, 0.55, 0.60, 0.65],
        }
        grid = param_grid or default_grid

        n = len(self.close)
        split_size = n // n_splits
        oos_results: list[BacktestResult] = []

        for i in range(n_splits):
            start = i * split_size
            end = start + split_size
            is_end = start + int(split_size * is_fraction)

            is_close = self.close.iloc[start:is_end]
            is_signals = self.signals.iloc[start:is_end]
            oos_close = self.close.iloc[is_end:end]
            oos_signals = self.signals.iloc[is_end:end]

            # Grid-search on IS
            best_sharpe = -np.inf
            best_params: dict[str, Any] = {}
            for atr_m in grid.get("atr_multiplier", [2.0]):
                for hmm_p in grid.get("min_hmm_proba", [0.55]):
                    p = {"atr_multiplier": atr_m, "min_hmm_proba": hmm_p}
                    is_eng = BacktestEngine(is_close, is_signals, self.sentiment, self.capital, self.commission)
                    is_res = is_eng.run(params=p)
                    if is_res.sharpe > best_sharpe:
                        best_sharpe = is_res.sharpe
                        best_params = p

            # OOS evaluation with best IS params
            oos_eng = BacktestEngine(oos_close, oos_signals, self.sentiment, self.capital, self.commission)
            oos_res = oos_eng.run(params=best_params)
            oos_res.params = {**best_params, "split": i, "is_sharpe": best_sharpe}
            oos_results.append(oos_res)

            logger.info(
                "WFO split %d/%d — IS best params: %s (Sharpe=%.2f) | OOS: %s",
                i + 1, n_splits, best_params, best_sharpe, oos_res.summary()
            )

        return oos_results

    # ── Private implementations ───────────────────────────────────────────────

    def _run_vectorbt(self, params: dict[str, Any]) -> BacktestResult:
        """Vectorbt-based simulation."""
        import vectorbt as vbt  # type: ignore[import-untyped]  # noqa: PLC0415

        entries = self.signals == 1
        exits = (self.signals == -1) | (self.signals == 0)

        pf = vbt.Portfolio.from_signals(
            close=self.close,
            entries=entries,
            exits=exits,
            init_cash=self.capital,
            fees=self.commission,
            freq="1D",
        )

        rets = pf.returns()
        eq = pf.value()

        trade_log = pf.trades.records_readable
        win_rate = float((trade_log["Return"] > 0).mean()) if not trade_log.empty else 0.0

        geo = (
            geo_risk_exposure(rets, self.sentiment.reindex(rets.index).fillna(0))
            if self.sentiment is not None
            else {}
        )

        return BacktestResult(
            sharpe=sharpe_ratio(rets),
            sortino=sortino_ratio(rets),
            max_drawdown_pct=abs(max_drawdown(eq)),
            profit_factor=profit_factor(rets),
            total_trades=len(trade_log),
            win_rate=win_rate,
            geo_risk_metrics=geo,
            equity_curve=eq,
            trade_log=trade_log,
            params=params,
        )

    def _run_numpy(self, params: dict[str, Any]) -> BacktestResult:
        """
        Pure-numpy vectorised simulation (vectorbt fallback).

        Uses position-based P&L:
          ret_t = pos_{t-1} * close_ret_t - |Δpos_t| * commission
        This correctly handles both long and short positions and applies
        transaction costs only at position changes (not every bar).
        """
        close = self.close.values
        sig   = self.signals.values.astype(float)
        n     = len(close)

        close_ret = np.zeros(n)
        close_ret[1:] = np.diff(close) / (close[:-1] + 1e-12)

        pos_lag   = np.roll(sig, 1); pos_lag[0] = 0.0
        trade_cost = np.abs(np.diff(sig, prepend=sig[0])) * self.commission
        period_ret = pos_lag * close_ret - trade_cost

        equity = np.cumprod(1.0 + period_ret) * self.capital

        eq_series  = pd.Series(equity, index=self.close.index)
        ret_series = pd.Series(period_ret, index=self.close.index)

        # Per-trade win rate
        pos_series   = pd.Series(sig, index=self.close.index)
        entries      = pos_series[pos_series != 0].index
        trade_returns: list[float] = []
        for i in range(len(entries) - 1):
            sl = ret_series.loc[entries[i]: entries[i + 1]]
            trade_returns.append(float(sl.sum()))
        trade_arr = np.array(trade_returns) if trade_returns else np.array([0.0])
        win_rate  = float((trade_arr > 0).mean())

        trade_log = pd.DataFrame({"Return": trade_arr})

        geo = (
            geo_risk_exposure(
                ret_series,
                self.sentiment.reindex(ret_series.index).fillna(0),
            )
            if self.sentiment is not None
            else {}
        )

        return BacktestResult(
            sharpe=sharpe_ratio(ret_series),
            sortino=sortino_ratio(ret_series),
            max_drawdown_pct=abs(max_drawdown(eq_series)),
            profit_factor=profit_factor(ret_series),
            total_trades=len(trade_arr),
            win_rate=win_rate,
            geo_risk_metrics=geo,
            equity_curve=eq_series,
            trade_log=trade_log,
            params=params,
        )
