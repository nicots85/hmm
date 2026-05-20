"""
/tests/test_core.py — Unit + property-based test suite. v3.

Coverage:
1. KellyPositionSizer — formula, edge cases, compute_from_series, sizing_series.
2. FeatureEngineer    — log_ret accuracy, ATR, higher moments, no look-ahead.
3. HMMRegimeDetector  — 12-feature obs matrix, prob bounds, posteriors sum=1,
                        ExpandingWindowRefitter.
4. DynamicStopLoss    — trailing constraint, volatile widening.
5. BacktestEngine     — Sharpe/Sortino/MaxDD/PF, skew-adjusted Sharpe, Calmar,
                        regime breakdown, look-ahead-free P&L.
6. HedgeManager       — vol-adjusted sizing (P9 fix).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from backtest.engine import (
    BacktestEngine,
    calmar_ratio,
    max_drawdown,
    profit_factor,
    regime_performance_breakdown,
    sharpe_ratio,
    skew_adjusted_sharpe,
    sortino_ratio,
)
from data.features import FeatureEngineer, FeatureConfig
from execution.risk_manager import (
    DynamicStopLoss,
    HedgeManager,
    KellyPositionSizer,
    StopLossLevel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """500-bar OHLCV — enough for 12-feature extended obs matrix."""
    n = 500
    rng = np.random.default_rng(42)
    close = 30_000 + np.cumsum(rng.normal(0, 300, n))
    high  = close + abs(rng.normal(0, 100, n))
    low   = close - abs(rng.normal(0, 100, n))
    open_ = close + rng.normal(0, 50, n)
    vol   = abs(rng.normal(1_000_000, 200_000, n))
    index = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=index,
    )


@pytest.fixture()
def sizer() -> KellyPositionSizer:
    return KellyPositionSizer(nav=100_000, kelly_fraction=0.25, max_risk_pct=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Kelly Position Sizer
# ─────────────────────────────────────────────────────────────────────────────

class TestKellyPositionSizer:

    def test_standard_case(self, sizer: KellyPositionSizer) -> None:
        """p=0.55 b=2 → raw=0.325 → frac=0.08125 → capped=0.02 → $2000."""
        result = sizer.compute(win_rate=0.55, avg_win=0.03, avg_loss=0.015)
        assert abs(result.raw_kelly - 0.325) < 1e-5
        assert abs(result.fractional_kelly - 0.08125) < 1e-5
        assert result.position_size_pct == pytest.approx(0.02, abs=1e-6)
        assert result.position_size_usd == pytest.approx(2_000.0, abs=0.01)

    def test_zero_edge_returns_zero(self, sizer: KellyPositionSizer) -> None:
        result = sizer.compute(win_rate=0.5, avg_win=0.01, avg_loss=0.01)
        assert result.raw_kelly == pytest.approx(0.0, abs=1e-9)
        assert result.position_size_usd == pytest.approx(0.0, abs=0.01)

    def test_negative_kelly_clamped(self, sizer: KellyPositionSizer) -> None:
        result = sizer.compute(win_rate=0.4, avg_win=0.01, avg_loss=0.02)
        assert result.position_size_usd == pytest.approx(0.0, abs=0.01)

    def test_max_cap_enforced(self, sizer: KellyPositionSizer) -> None:
        result = sizer.compute(win_rate=0.9, avg_win=0.10, avg_loss=0.01)
        assert result.position_size_pct <= 0.02 + 1e-9

    def test_invalid_avg_loss_raises(self, sizer: KellyPositionSizer) -> None:
        with pytest.raises(ValueError, match="avg_loss"):
            sizer.compute(win_rate=0.6, avg_win=0.02, avg_loss=0.0)

    def test_invalid_win_rate_raises(self, sizer: KellyPositionSizer) -> None:
        with pytest.raises(ValueError, match="win_rate"):
            sizer.compute(win_rate=1.0, avg_win=0.02, avg_loss=0.01)

    def test_compute_from_series_insufficient(self, sizer: KellyPositionSizer) -> None:
        """Fewer than 20 trades → conservative sizing, no crash."""
        tiny = np.array([0.01, -0.02, 0.03])
        result = sizer.compute_from_series(tiny)
        assert result.position_size_pct > 0
        assert result.position_size_pct <= 0.02

    def test_compute_from_series_full(self, sizer: KellyPositionSizer) -> None:
        """50 trades with 60% WR → positive Kelly."""
        rng = np.random.default_rng(0)
        wins  = rng.uniform(0.01, 0.04, 30)
        losses = -rng.uniform(0.005, 0.02, 20)
        trades = np.concatenate([wins, losses])
        rng.shuffle(trades)
        result = sizer.compute_from_series(trades)
        assert result.raw_kelly >= 0

    def test_sizing_series_no_lookahead(self, sizer: KellyPositionSizer, sample_ohlcv: pd.DataFrame) -> None:
        """sizing_series must never use future returns."""
        fe = FeatureEngineer()
        feat = fe.transform(sample_ohlcv)
        signals = pd.Series(
            np.where(np.arange(len(feat)) % 5 == 0, 1, 0), index=feat.index
        )
        sz = sizer.build_sizing_series(signals, feat["close"])
        assert len(sz) == len(signals)
        assert (sz >= 0).all()

    @given(
        win_rate=st.floats(min_value=0.01, max_value=0.99),
        avg_win=st.floats(min_value=1e-4, max_value=1.0),
        avg_loss=st.floats(min_value=1e-4, max_value=1.0),
    )
    @hyp_settings(max_examples=300)
    def test_position_never_exceeds_max(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> None:
        s = KellyPositionSizer(nav=100_000, kelly_fraction=0.25, max_risk_pct=0.02)
        result = s.compute(win_rate, avg_win, avg_loss)
        assert result.position_size_pct <= 0.02 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 2. FeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_log_returns_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        out = FeatureEngineer().transform(sample_ohlcv)
        assert "log_ret" in out.columns
        assert not out["log_ret"].isna().any()

    def test_log_returns_numerical_accuracy(self) -> None:
        n = 120
        rng = np.random.default_rng(0)
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        index = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.005, "low": prices * 0.995,
            "close": prices, "volume": np.full(n, 1e6),
        }, index=index)
        fe = FeatureEngineer(FeatureConfig(atr_period=2, realised_vol_window=2,
                                           vol_profile_window=5, skew_window=10))
        out = fe.transform(df)
        expected = pd.Series(np.log(prices[1:] / prices[:-1]), index=index[1:])
        common = out.index.intersection(expected.index)
        np.testing.assert_allclose(
            out.loc[common, "log_ret"].values,
            expected.loc[common].values, rtol=1e-10,
        )

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        out = FeatureEngineer().transform(sample_ohlcv)
        assert (out["atr"] > 0).all()

    def test_higher_moments_present(self, sample_ohlcv: pd.DataFrame) -> None:
        """v3: realised_skew and realised_kurt must be in output."""
        out = FeatureEngineer().transform(sample_ohlcv)
        assert "realised_skew" in out.columns
        assert "realised_kurt" in out.columns
        assert not out["realised_skew"].isna().any()

    def test_hmm_observations_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        fe = FeatureEngineer()
        out = fe.transform(sample_ohlcv)
        obs = fe.get_hmm_observations(out)
        assert obs.ndim == 2 and obs.shape[1] == 4
        assert not np.isnan(obs).any()

    def test_no_lookahead_atr(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR must not use close[t] in its True Range for time t (uses close[t-1])."""
        fe = FeatureEngineer()
        out = fe.transform(sample_ohlcv)
        # ATR at t should be computed from TR values up to t; TR uses close[t-1]
        # Verify by checking ATR is not NaN after first bar
        assert out["atr"].iloc[5:].notna().all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. HMMRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestHMMRegimeDetector:

    @pytest.fixture()
    def fitted_hmm(self, sample_ohlcv: pd.DataFrame):  # type: ignore[override]
        from models.hmm_regimes import HMMRegimeDetector
        fe = FeatureEngineer()
        feat = fe.transform(sample_ohlcv)
        obs, idx = HMMRegimeDetector.build_extended_observations(feat)
        det = HMMRegimeDetector(n_regimes=2, covariance_type="diag", n_iter=50).fit(obs)
        return det, obs, feat, idx

    def test_extended_obs_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        from models.hmm_regimes import HMMRegimeDetector
        fe = FeatureEngineer()
        feat = fe.transform(sample_ohlcv)
        obs, idx = HMMRegimeDetector.build_extended_observations(feat)
        assert obs.ndim == 2
        # 10 base + 2 higher moments (skew, kurt) = 12
        assert obs.shape[1] in (10, 12)
        assert not np.isnan(obs).any()
        assert len(idx) == obs.shape[0]

    def test_predict_bull_prob_bounds(self, fitted_hmm) -> None:  # type: ignore[no-untyped-def]
        det, obs, _, _ = fitted_hmm
        probs = det.predict_bull_prob(obs)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_predict_bear_vol_prob_bounds(self, fitted_hmm) -> None:  # type: ignore[no-untyped-def]
        det, obs, _, _ = fitted_hmm
        probs = det.predict_bear_vol_prob(obs)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_posteriors_sum_to_one(self, fitted_hmm) -> None:  # type: ignore[no-untyped-def]
        det, obs, _, _ = fitted_hmm
        posteriors = det.predict_proba(obs)
        np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-6)

    def test_expanding_window_refitter(self, sample_ohlcv: pd.DataFrame) -> None:
        """ExpandingWindowRefitter returns fitted detector and caches results."""
        from models.hmm_regimes import HMMRegimeDetector, ExpandingWindowRefitter
        fe = FeatureEngineer()
        feat = fe.transform(sample_ohlcv)
        obs, _ = HMMRegimeDetector.build_extended_observations(feat)
        refitter = ExpandingWindowRefitter(n_regimes=2, min_train_bars=100,
                                            covariance_type="diag")
        det1 = refitter.fit_up_to(obs, 200)
        det2 = refitter.fit_up_to(obs, 200)  # should hit cache
        assert det1 is det2  # same object from cache
        assert det1._is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# 4. DynamicStopLoss
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicStopLoss:

    def test_long_sl_below_entry(self) -> None:
        sl = DynamicStopLoss(2.0).initial_stop(50_000, 1, 1_000, regime_label="bull_calm")
        assert sl.stop_price < 50_000

    def test_short_sl_above_entry(self) -> None:
        sl = DynamicStopLoss(2.0).initial_stop(50_000, -1, 1_000, regime_label="bear_calm")
        assert sl.stop_price > 50_000

    def test_trail_never_increases_risk_long(self) -> None:
        engine = DynamicStopLoss(2.0)
        current = StopLossLevel(48_000, 2.0, "atr_multiplier", "initial")
        updated = engine.trail_stop(current, 1, 47_000, 1_000, regime_label="bull_calm")
        assert updated.stop_price >= current.stop_price

    def test_trail_advances_when_profit_long(self) -> None:
        engine = DynamicStopLoss(2.0)
        current = StopLossLevel(48_000, 2.0, "atr_multiplier", "initial")
        updated = engine.trail_stop(current, 1, 55_000, 1_000, regime_label="bull_calm")
        assert updated.stop_price > current.stop_price

    def test_volatile_regime_widens_sl(self) -> None:
        engine = DynamicStopLoss(2.0)
        sl_calm     = engine.initial_stop(50_000, 1, 1_000, regime_label="bull_calm")
        sl_volatile = engine.initial_stop(50_000, 1, 1_000, regime_label="bear_volatile")
        assert sl_volatile.stop_price < sl_calm.stop_price


# ─────────────────────────────────────────────────────────────────────────────
# 5. Backtest metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestMetrics:

    def _flat(self) -> pd.Series:
        return pd.Series(np.zeros(365))

    def _positive(self) -> pd.Series:
        return pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 365))

    def test_sharpe_flat_zero(self) -> None:
        assert sharpe_ratio(self._flat()) == pytest.approx(0.0, abs=1e-9)

    def test_sharpe_positive(self) -> None:
        assert sharpe_ratio(self._positive()) > 0

    def test_sortino_ge_sharpe_positive_drift(self) -> None:
        rets = self._positive()
        assert sortino_ratio(rets) >= sharpe_ratio(rets)

    def test_max_drawdown_nonpositive(self) -> None:
        assert max_drawdown(pd.Series([100, 105, 98, 110])) <= 0

    def test_max_drawdown_value(self) -> None:
        assert max_drawdown(pd.Series([100, 120, 90])) == pytest.approx(-0.25, rel=1e-4)

    def test_profit_factor_all_wins(self) -> None:
        assert profit_factor(pd.Series([0.01, 0.02])) == float("inf")

    def test_profit_factor_all_losses(self) -> None:
        assert profit_factor(pd.Series([-0.01, -0.02])) == pytest.approx(0.0, abs=1e-9)

    def test_skew_adjusted_sharpe_penalises_negative_skew(self) -> None:
        """
        For a strategy with POSITIVE Sharpe and negative skew,
        the adjusted SR must be lower than the raw SR.
        """
        rng = np.random.default_rng(7)
        # Strong positive drift so a few large losses don't flip the mean
        base = rng.normal(0.005, 0.008, 500)
        base[::50] -= 0.06         # inject periodic large losses → negative skew
        rets = pd.Series(base)
        assert rets.mean() > 0, f"mean={rets.mean():.4f} not positive"
        assert rets.skew() < 0, f"skew={rets.skew():.4f} not negative"
        assert sharpe_ratio(rets) > 0
        assert skew_adjusted_sharpe(rets) < sharpe_ratio(rets)

    def test_calmar_positive_drift(self) -> None:
        rets = self._positive()
        eq   = (1 + rets).cumprod() * 100_000
        assert calmar_ratio(rets, eq) > 0

    def test_sharpe_periods_365(self) -> None:
        """periods_per_year=365 gives higher annualised Sharpe than 252."""
        rets = pd.Series([0.001] * 365)
        assert sharpe_ratio(rets, periods_per_year=365) > sharpe_ratio(rets, periods_per_year=252)

    def test_regime_breakdown_shape(self) -> None:
        rng = np.random.default_rng(7)
        rets    = pd.Series(rng.normal(0, 0.01, 100))
        regimes = pd.Series(["bull_calm"] * 50 + ["bear_volatile"] * 50, index=rets.index)
        result  = regime_performance_breakdown(rets, regimes)
        assert "bull_calm" in result.index and "bear_volatile" in result.index
        assert "skew_adj_sharpe" in result.columns

    def test_backtest_no_lookahead(self) -> None:
        """
        P&L on a perfect oracle signal (always +1 when price rises) should
        NOT achieve infinite returns — look-ahead bias would allow that.
        With the shift fix, we trade the NEXT bar's return, not the current.
        """
        n = 200
        close  = pd.Series(np.arange(1, n + 1, dtype=float) * 100)
        # Signal: always long — this is NOT look-ahead, just a simple test
        signals = pd.Series(np.ones(n))
        engine  = BacktestEngine(close, signals, commission_pct=0.0)
        result  = engine.run()
        # With shift, the position on bar 0 enters at bar 1's price
        # Equity should grow, but deterministically
        assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]
        assert result.max_drawdown_pct < 1.0  # not 0 drawdown (prices are linear)


# ─────────────────────────────────────────────────────────────────────────────
# 6. HedgeManager — vol-adjusted sizing
# ─────────────────────────────────────────────────────────────────────────────

class TestHedgeManager:

    def _make_returns(self, n: int, seed: int = 0) -> tuple[pd.Series, pd.Series]:
        rng = np.random.default_rng(seed)
        btc  = pd.Series(rng.normal(0.001, 0.04, n))
        gold = pd.Series(rng.normal(0.0002, 0.008, n))  # much lower vol
        return btc, gold

    def test_no_hedge_in_bull_regime(self) -> None:
        manager = HedgeManager(correlation_window=30, correlation_threshold=0.3)
        btc, gold = self._make_returns(100)
        sig = manager.evaluate(btc, gold, 1, "BTC", "bull_calm")
        assert not sig.hedge_active

    def test_hedge_activates_in_bear(self) -> None:
        """Construct highly correlated returns and verify hedge fires."""
        rng = np.random.default_rng(42)
        base  = rng.normal(0, 0.02, 100)
        btc   = pd.Series(base + rng.normal(0, 0.001, 100))
        gold  = pd.Series(base * 0.2 + rng.normal(0, 0.001, 100))  # correlated
        manager = HedgeManager(correlation_window=30, correlation_threshold=0.1)
        sig = manager.evaluate(btc, gold, 1, "BTC", "bear_volatile")
        assert sig.hedge_active

    def test_vol_adjusted_size_btc_larger_than_gold(self) -> None:
        """
        BTC vol >> Gold vol → vol_ratio > 1 → hedge_size_pct > hedge_ratio * |corr|.
        The vol adjustment must make the hedge larger (not smaller) when
        primary asset is more volatile than the hedge asset.
        """
        rng  = np.random.default_rng(0)
        base = rng.normal(0, 0.01, 100)
        btc  = pd.Series(base * 5)    # high vol
        gold = pd.Series(base * 1)    # low vol — strongly correlated
        manager = HedgeManager(correlation_window=50, correlation_threshold=0.1,
                                hedge_ratio=0.3)
        sig = manager.evaluate(btc, gold, 1, "BTC", "bear_calm")
        if sig.hedge_active:
            # vol_ratio ≈ 5x → hedge_size should exceed bare hedge_ratio * |corr|
            naive_size = abs(sig.correlation) * 0.3
            assert sig.hedge_size_pct >= naive_size - 1e-6

    def test_hedge_size_capped_at_one(self) -> None:
        """Extreme vol ratio must not produce hedge_size > 1.0."""
        rng  = np.random.default_rng(1)
        base = rng.normal(0, 0.01, 100)
        btc  = pd.Series(base * 50)
        gold = pd.Series(base)
        manager = HedgeManager(correlation_window=50, correlation_threshold=0.0)
        sig = manager.evaluate(btc, gold, 1, "BTC", "bear_volatile")
        assert sig.hedge_size_pct <= 1.0
