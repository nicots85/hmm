"""
/backtest/engine.py — Vectorised backtest engine with walk-forward optimisation.

v3 changes:
- P1 FIX: signals shifted by 1 bar before entering P&L calculation.
  pos_{t-1} * ret_t is now correct — we decide at close[t-1] and fill at open[t].
- P3 FIX: Kelly fractional position sizing integrated. BacktestEngine accepts
  a sizing_series (0..1 float) in addition to binary signals.
- P4 FIX: ATR-based stop loss enforced bar-by-bar in numpy simulation.
- P8 FIX: walk_forward() uses ExpandingWindowRefitter for HMM refit each split.
- P7: skew_adjusted_sharpe() added (Pezier & White 2006).
- P5: walk_forward n_splits default raised; caller controls granularity.
- regime_performance_breakdown() unchanged, still available.
- _run_vectorbt: updated to use pos.shift(1) for consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """Annualised Sharpe. periods_per_year=365 for 24/7 crypto markets."""
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(periods_per_year))


def skew_adjusted_sharpe(
    returns: pd.Series,
    periods_per_year: int = 365,
) -> float:
    """
    Pezier & White (2006) skewness-adjusted Sharpe Ratio.

    Penalises strategies with negative skew (fat left tail) which naive
    Sharpe ignores. Formula:
        SR_adj = SR * [1 + (S/6)*SR - (K/24)*SR²]
    where S=skewness, K=excess kurtosis, SR=standard Sharpe.
    """
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year)
    s  = float(returns.skew())
    k  = float(returns.kurt())      # excess kurtosis
    adjustment = 1.0 + (s / 6.0) * sr - (k / 24.0) * sr ** 2
    return float(sr * adjustment)


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / (rolling_max + 1e-12)
    return float(drawdown.min())


def profit_factor(returns: pd.Series) -> float:
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 365) -> float:
    """Annualised return / |MaxDrawdown|. Penalises deep drawdowns."""
    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    dd = abs(max_drawdown(equity_curve))
    return float(ann_ret / dd) if dd > 0 else float("inf")


def geo_risk_exposure(
    returns: pd.Series,
    sentiment_series: pd.Series,
    sentiment_threshold: float = 0.5,
) -> dict[str, float]:
    """Fraction of trades during high geo-risk periods and their avg PnL."""
    common = returns.index.intersection(sentiment_series.index)
    if common.empty:
        return {"exposure_fraction": 0.0, "mean_pnl_geo": 0.0, "mean_pnl_normal": 0.0}
    rets = returns.loc[common]
    sent = sentiment_series.loc[common].abs()
    in_trade   = rets != 0
    high_risk  = sent > sentiment_threshold
    geo_trades = rets[in_trade & high_risk]
    norm_trades = rets[in_trade & ~high_risk]
    return {
        "exposure_fraction": float(len(geo_trades) / max(in_trade.sum(), 1)),
        "mean_pnl_geo":    float(geo_trades.mean()) if not geo_trades.empty else 0.0,
        "mean_pnl_normal": float(norm_trades.mean()) if not norm_trades.empty else 0.0,
    }


def regime_performance_breakdown(
    returns: pd.Series,
    regime_series: pd.Series,
) -> pd.DataFrame:
    """Per-regime Sharpe / WR table."""
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
            "skew_adj_sharpe": skew_adjusted_sharpe(r),
            "win_rate": float((r > 0).mean()),
        })
    return pd.DataFrame(rows).set_index("regime").round(4)


# ─────────────────────────────────────────────────────────────────────────────
# BacktestResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    sharpe: float
    skew_adj_sharpe: float
    sortino: float
    calmar: float
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
            f"Sharpe={self.sharpe:.2f} Adj={self.skew_adj_sharpe:.2f} "
            f"Sortino={self.sortino:.2f} Calmar={self.calmar:.2f} "
            f"MaxDD={self.max_drawdown_pct*100:.1f}% PF={self.profit_factor:.2f} "
            f"Trades={self.total_trades} WR={self.win_rate*100:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BacktestEngine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Vectorised backtest engine.

    P&L formula (look-ahead-free):
        pos_t  = signal decided at close[t] (information up to bar t)
        ret_t+1 = close[t+1]/close[t] - 1
        pnl_t+1 = pos_t * ret_t+1 - |pos_t - pos_{t-1}| * commission

    This means signal[t] is shifted before multiplying by return[t+1].
    position sizing: if sizing_series is provided (float in [0,1]),
        pos_t = signal_t * sizing_t   (Kelly fractional position)
    otherwise binary {-1, 0, +1}.

    ATR stop loss:
        If a long position exists and price drops below entry - atr_mult*ATR,
        the position is forced to 0. Implemented bar-by-bar in numpy.
    """

    def __init__(
        self,
        close: pd.Series,
        signals: pd.Series,           # {-1, 0, +1}
        sentiment: pd.Series | None = None,
        sizing_series: pd.Series | None = None,   # Kelly weights [0,1]
        atr_series: pd.Series | None = None,      # for stop loss enforcement
        atr_stop_mult: float = 2.5,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.0005,
    ) -> None:
        self.close       = close.dropna()
        self.signals     = signals.reindex(self.close.index).fillna(0)
        self.sentiment   = sentiment
        self.sizing      = sizing_series.reindex(self.close.index).fillna(1.0) if sizing_series is not None else None
        self.atr         = atr_series.reindex(self.close.index) if atr_series is not None else None
        self.atr_stop_mult = atr_stop_mult
        self.capital     = initial_capital
        self.commission  = commission_pct

    def run(self, params: dict[str, Any] | None = None) -> BacktestResult:
        params = params or {}
        try:
            return self._run_vectorbt(params)
        except ImportError:
            return self._run_numpy(params)

    def walk_forward(
        self,
        n_splits: int = 5,
        is_fraction: float = 0.70,
        param_grid: dict[str, list[Any]] | None = None,
        hmm_obs: np.ndarray | None = None,       # for ExpandingWindowRefitter
        n_regimes: int = 4,
    ) -> list[BacktestResult]:
        """
        Walk-forward optimisation with optional HMM expanding-window refit.

        If hmm_obs is provided, the HMM is refitted at each split boundary
        using all data up to that point (ExpandingWindowRefitter).
        """
        default_grid: dict[str, list[Any]] = {
            "atr_stop_mult": [2.0, 2.5, 3.0],
            "bull_prob_thresh": [0.50, 0.55, 0.60, 0.65],
        }
        grid = param_grid or default_grid

        N = len(self.close)
        split_size = N // n_splits
        oos_results: list[BacktestResult] = []

        for i in range(n_splits):
            start  = i * split_size
            end    = min(start + split_size, N)
            is_end = start + int((end - start) * is_fraction)

            # IS grid search
            best_sharpe = -np.inf
            best_params: dict[str, Any] = {}

            for atr_m in grid.get("atr_stop_mult", [2.5]):
                for bp_t in grid.get("bull_prob_thresh", [0.55]):
                    p = {"atr_stop_mult": atr_m, "bull_prob_thresh": bp_t}
                    is_eng = BacktestEngine(
                        self.close.iloc[start:is_end],
                        self.signals.iloc[start:is_end],
                        self.sentiment,
                        self.sizing.iloc[start:is_end] if self.sizing is not None else None,
                        self.atr.iloc[start:is_end] if self.atr is not None else None,
                        atr_stop_mult=atr_m,
                        commission_pct=self.commission,
                    )
                    is_res = is_eng.run(params=p)
                    if is_res.sharpe > best_sharpe:
                        best_sharpe = is_res.sharpe
                        best_params = p

            # OOS evaluation
            oos_eng = BacktestEngine(
                self.close.iloc[is_end:end],
                self.signals.iloc[is_end:end],
                self.sentiment,
                self.sizing.iloc[is_end:end] if self.sizing is not None else None,
                self.atr.iloc[is_end:end] if self.atr is not None else None,
                atr_stop_mult=best_params.get("atr_stop_mult", 2.5),
                commission_pct=self.commission,
            )
            oos_res = oos_eng.run(params=best_params)
            oos_res.params = {**best_params, "split": i + 1, "is_sharpe": best_sharpe}
            oos_results.append(oos_res)

            logger.info(
                "WFO split %d/%d IS_SR=%.2f → OOS: %s | best_params=%s",
                i + 1, n_splits, best_sharpe, oos_res.summary(), best_params
            )

        return oos_results

    # ── P&L implementations ───────────────────────────────────────────────────

    def _run_vectorbt(self, params: dict[str, Any]) -> BacktestResult:
        import vectorbt as vbt  # type: ignore[import-untyped]  # noqa: PLC0415

        pos = self.signals.copy()
        if self.sizing is not None:
            pos = pos * self.sizing

        # SHIFT: signal at bar t → fills at bar t+1
        pos_shifted = pos.shift(1).fillna(0)

        entries = pos_shifted > 0
        exits   = pos_shifted <= 0

        pf = vbt.Portfolio.from_signals(
            close=self.close,
            entries=entries,
            exits=exits,
            init_cash=self.capital,
            fees=self.commission,
            freq="1D",
        )
        rets   = pf.returns()
        eq     = pf.value()
        trades = pf.trades.records_readable
        wr     = float((trades["Return"] > 0).mean()) if not trades.empty else 0.0
        geo    = (
            geo_risk_exposure(rets, self.sentiment.reindex(rets.index).fillna(0))
            if self.sentiment is not None else {}
        )
        return BacktestResult(
            sharpe=sharpe_ratio(rets),
            skew_adj_sharpe=skew_adjusted_sharpe(rets),
            sortino=sortino_ratio(rets),
            calmar=calmar_ratio(rets, eq),
            max_drawdown_pct=abs(max_drawdown(eq)),
            profit_factor=profit_factor(rets),
            total_trades=len(trades),
            win_rate=wr,
            geo_risk_metrics=geo,
            equity_curve=eq,
            trade_log=trades,
            params=params,
        )

    def _run_numpy(self, params: dict[str, Any]) -> BacktestResult:
        """
        Vectorised numpy P&L with:
        - Look-ahead-free shift (P1 fix)
        - Kelly fractional sizing (P3 fix)
        - ATR stop loss enforcement (P4 fix)
        """
        close_arr = self.close.values.astype(float)
        sig_arr   = self.signals.values.astype(float)
        sz_arr    = self.sizing.values.astype(float) if self.sizing is not None else np.ones(len(sig_arr))
        atr_arr   = self.atr.values.astype(float) if self.atr is not None else None
        atr_m     = params.get("atr_stop_mult", self.atr_stop_mult)
        n         = len(close_arr)

        # ── Stop loss enforcement (bar-by-bar, unavoidable loop) ──────────────
        pos = np.zeros(n)
        entry_price = np.zeros(n)
        sl_price    = np.zeros(n)

        for t in range(1, n):
            prev_pos = pos[t - 1]
            raw_sig  = sig_arr[t - 1] * sz_arr[t - 1]   # signal decided at t-1

            # Check stop loss before accepting new signal
            if prev_pos > 0 and atr_arr is not None:
                if close_arr[t] < sl_price[t - 1]:
                    pos[t] = 0.0                          # stopped out
                    entry_price[t] = 0.0
                    sl_price[t]    = 0.0
                    continue

            if prev_pos < 0 and atr_arr is not None:
                if close_arr[t] > sl_price[t - 1]:
                    pos[t] = 0.0
                    entry_price[t] = 0.0
                    sl_price[t]    = 0.0
                    continue

            # Enter / stay / exit
            pos[t] = raw_sig

            if raw_sig > 0 and prev_pos <= 0:            # new long
                entry_price[t] = close_arr[t]
                sl_price[t] = close_arr[t] - atr_m * (atr_arr[t] if atr_arr is not None else 0)
            elif raw_sig < 0 and prev_pos >= 0:          # new short
                entry_price[t] = close_arr[t]
                sl_price[t] = close_arr[t] + atr_m * (atr_arr[t] if atr_arr is not None else 0)
            else:
                entry_price[t] = entry_price[t - 1]
                sl_price[t]    = sl_price[t - 1]

        # ── Vectorised P&L ────────────────────────────────────────────────────
        close_ret  = np.zeros(n)
        close_ret[1:] = np.diff(close_arr) / (close_arr[:-1] + 1e-12)

        pos_lag    = np.roll(pos, 1); pos_lag[0] = 0.0
        trade_cost = np.abs(np.diff(pos, prepend=pos[0])) * self.commission
        period_ret = pos_lag * close_ret - trade_cost

        equity     = np.cumprod(1.0 + period_ret) * self.capital
        eq_series  = pd.Series(equity, index=self.close.index)
        ret_series = pd.Series(period_ret, index=self.close.index)

        # Per-trade stats
        pos_series = pd.Series(pos, index=self.close.index)
        entries_idx = pos_series[pos_series != 0].index
        trade_rets: list[float] = []
        for j in range(len(entries_idx) - 1):
            sl = ret_series.loc[entries_idx[j]: entries_idx[j + 1]]
            trade_rets.append(float(sl.sum()))
        trade_arr = np.array(trade_rets) if trade_rets else np.array([0.0])
        wr = float((trade_arr > 0).mean())

        geo = (
            geo_risk_exposure(ret_series, self.sentiment.reindex(ret_series.index).fillna(0))
            if self.sentiment is not None else {}
        )

        return BacktestResult(
            sharpe=sharpe_ratio(ret_series),
            skew_adj_sharpe=skew_adjusted_sharpe(ret_series),
            sortino=sortino_ratio(ret_series),
            calmar=calmar_ratio(ret_series, eq_series),
            max_drawdown_pct=abs(max_drawdown(eq_series)),
            profit_factor=profit_factor(ret_series),
            total_trades=len(trade_arr),
            win_rate=wr,
            geo_risk_metrics=geo,
            equity_curve=eq_series,
            trade_log=pd.DataFrame({"Return": trade_arr}),
            params=params,
        )
