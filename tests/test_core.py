"""
/tests/test_core.py — Unit tests for core quantitative functions.

Coverage:
1. KellyPositionSizer — correctness of fractional Kelly formula including
   edge cases (zero edge, max-cap enforcement, invalid inputs).
2. FeatureEngineer.log_returns — numerical accuracy and NaN handling.
3. DynamicStopLoss — trailing-only constraint (SL must never increase risk).
4. BacktestEngine metrics — Sharpe, Sortino, max_drawdown sanity checks.

Property-based tests use Hypothesis to find edge cases in the Kelly
calculator that deterministic tests might miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from backtest.engine import (
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from data.features import FeatureEngineer, FeatureConfig
from execution.risk_manager import DynamicStopLoss, KellyPositionSizer, StopLossLevel


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """500-bar OHLCV DataFrame — sufficient for extended HMM observation building."""
    n = 500
    rng = np.random.default_rng(42)
    close = 30_000 + np.cumsum(rng.normal(0, 300, n))
    high = close + abs(rng.normal(0, 100, n))
    low = close - abs(rng.normal(0, 100, n))
    open_ = close + rng.normal(0, 50, n)
    volume = abs(rng.normal(1_000_000, 200_000, n))

    index = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
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
        """
        With p=0.55, avg_win=0.03, avg_loss=0.015:
        b = 0.03/0.015 = 2.0
        raw Kelly = (0.55*2 - 0.45) / 2 = 0.325
        fractional (×0.25) = 0.08125
        capped at max_risk_pct = 0.02
        position_usd = 0.02 × 100_000 = 2_000
        """
        result = sizer.compute(win_rate=0.55, avg_win=0.03, avg_loss=0.015)
        assert abs(result.raw_kelly - 0.325) < 1e-5
        assert abs(result.fractional_kelly - 0.08125) < 1e-5
        # Must be capped at max_risk_pct=0.02
        assert result.position_size_pct == pytest.approx(0.02, abs=1e-6)
        assert result.position_size_usd == pytest.approx(2_000.0, abs=0.01)

    def test_zero_edge_returns_zero(self, sizer: KellyPositionSizer) -> None:
        """Kelly = 0 when win_rate = 0.5 and b = 1 (coin flip, no edge)."""
        result = sizer.compute(win_rate=0.5, avg_win=0.01, avg_loss=0.01)
        assert result.raw_kelly == pytest.approx(0.0, abs=1e-9)
        assert result.position_size_usd == pytest.approx(0.0, abs=0.01)

    def test_negative_kelly_clamped_to_zero(self, sizer: KellyPositionSizer) -> None:
        """Negative Kelly (edge against us) must produce 0 position."""
        result = sizer.compute(win_rate=0.4, avg_win=0.01, avg_loss=0.02)
        assert result.position_size_usd == pytest.approx(0.0, abs=0.01)

    def test_max_cap_enforced(self, sizer: KellyPositionSizer) -> None:
        """Even with large edge, position must not exceed max_risk_pct of NAV."""
        result = sizer.compute(win_rate=0.9, avg_win=0.10, avg_loss=0.01)
        assert result.position_size_pct <= 0.02 + 1e-9

    def test_invalid_avg_loss_raises(self, sizer: KellyPositionSizer) -> None:
        with pytest.raises(ValueError, match="avg_loss"):
            sizer.compute(win_rate=0.6, avg_win=0.02, avg_loss=0.0)

    def test_invalid_win_rate_raises(self, sizer: KellyPositionSizer) -> None:
        with pytest.raises(ValueError, match="win_rate"):
            sizer.compute(win_rate=1.0, avg_win=0.02, avg_loss=0.01)

    @given(
        win_rate=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        avg_win=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False),
        avg_loss=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False),
    )
    @hyp_settings(max_examples=300)
    def test_position_size_never_exceeds_max(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> None:
        """Property: fractional Kelly result is always ≤ max_risk_pct × NAV."""
        result = self.sizer_instance().compute(win_rate, avg_win, avg_loss)
        assert result.position_size_pct <= 0.02 + 1e-9

    @staticmethod
    def sizer_instance() -> KellyPositionSizer:
        return KellyPositionSizer(nav=100_000, kelly_fraction=0.25, max_risk_pct=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FeatureEngineer — log returns
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_log_returns_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        fe = FeatureEngineer()
        out = fe.transform(sample_ohlcv)
        assert "log_ret" in out.columns
        assert not out["log_ret"].isna().any()

    def test_log_returns_numerical_accuracy(self) -> None:
        """log_ret = ln(c_t / c_{t-1}); verify against manual calculation."""
        # Use 120 bars so rolling windows in transform() don't eat all rows.
        n = 120
        rng = np.random.default_rng(0)
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        index = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.005,
            "low": prices * 0.995, "close": prices,
            "volume": np.full(n, 1e6),
        }, index=index)

        fe = FeatureEngineer(FeatureConfig(
            atr_period=2, realised_vol_window=2, vol_profile_window=5
        ))
        out = fe.transform(df)

        # Reconstruct expected log-returns for the rows that survived dropna
        expected = np.log(prices[1:] / prices[:-1])
        expected_series = pd.Series(expected, index=index[1:])
        common = out.index.intersection(expected_series.index)
        np.testing.assert_allclose(
            out.loc[common, "log_ret"].values,
            expected_series.loc[common].values,
            rtol=1e-10,
        )

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        fe = FeatureEngineer()
        out = fe.transform(sample_ohlcv)
        assert (out["atr"] > 0).all()

    def test_hmm_observations_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        fe = FeatureEngineer()
        out = fe.transform(sample_ohlcv)
        obs = fe.get_hmm_observations(out)
        assert obs.ndim == 2
        assert obs.shape[1] == 4
        assert not np.isnan(obs).any()


# ─────────────────────────────────────────────────────────────────────────────
# 2b. HMMRegimeDetector — extended observations + new methods
# ─────────────────────────────────────────────────────────────────────────────

class TestHMMRegimeDetector:

    @pytest.fixture()
    def fitted_hmm_and_obs(self, sample_ohlcv: pd.DataFrame):  # type: ignore[override]
        """Returns (detector, obs_array) fitted on 200-bar sample."""
        from models.hmm_regimes import HMMRegimeDetector
        fe = FeatureEngineer()
        feat = fe.transform(sample_ohlcv)
        obs, idx = HMMRegimeDetector.build_extended_observations(feat)
        # diag covariance + n_regimes=2: avoids degenerate transitions on
        # short test datasets (200 bars, 10 features → 133 free params with full)
        det = HMMRegimeDetector(n_regimes=2, covariance_type="diag", n_iter=50).fit(obs)
        return det, obs, feat, idx

    def test_extended_observations_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        """build_extended_observations returns (T, 10) matrix with no NaNs."""
        from models.hmm_regimes import HMMRegimeDetector
        fe = FeatureEngineer()
        feat = fe.transform(sample_ohlcv)
        obs, idx = HMMRegimeDetector.build_extended_observations(feat)
        assert obs.ndim == 2
        assert obs.shape[1] == 10
        assert not np.isnan(obs).any()
        assert len(idx) == obs.shape[0]

    def test_predict_bull_prob_bounds(self, fitted_hmm_and_obs) -> None:  # type: ignore[no-untyped-def]
        """predict_bull_prob values must be in [0, 1]."""
        det, obs, _, _ = fitted_hmm_and_obs
        probs = det.predict_bull_prob(obs)
        assert probs.shape == (len(obs),)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_bear_vol_prob_bounds(self, fitted_hmm_and_obs) -> None:  # type: ignore[no-untyped-def]
        """predict_bear_vol_prob values must be in [0, 1]."""
        det, obs, _, _ = fitted_hmm_and_obs
        probs = det.predict_bear_vol_prob(obs)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_bull_plus_other_probs_sum_to_one(self, fitted_hmm_and_obs) -> None:  # type: ignore[no-untyped-def]
        """Sum of all posterior probabilities per time step = 1."""
        det, obs, _, _ = fitted_hmm_and_obs
        posteriors = det.predict_proba(obs)
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DynamicStopLoss — trailing constraint
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicStopLoss:

    def test_long_sl_below_entry(self) -> None:
        sl_engine = DynamicStopLoss(atr_multiplier=2.0)
        sl = sl_engine.initial_stop(
            entry_price=50_000.0, direction=1,
            current_atr=1_000.0, regime_label="bull_calm"
        )
        assert sl.stop_price < 50_000.0

    def test_short_sl_above_entry(self) -> None:
        sl_engine = DynamicStopLoss(atr_multiplier=2.0)
        sl = sl_engine.initial_stop(
            entry_price=50_000.0, direction=-1,
            current_atr=1_000.0, regime_label="bear_calm"
        )
        assert sl.stop_price > 50_000.0

    def test_trail_never_increases_risk_long(self) -> None:
        """For a long, trailing must never LOWER the stop price."""
        sl_engine = DynamicStopLoss(atr_multiplier=2.0)
        current_sl = StopLossLevel(
            stop_price=48_000.0, distance_atr=2.0,
            anchor_type="atr_multiplier", rationale="initial"
        )
        # Simulate price falling — new proposed stop would be lower
        updated_sl = sl_engine.trail_stop(
            current_stop=current_sl,
            direction=1,
            current_price=47_000.0,   # price dropped
            current_atr=1_000.0,
            regime_label="bull_calm"
        )
        # Must keep the original (higher) stop
        assert updated_sl.stop_price >= current_sl.stop_price

    def test_trail_advances_when_price_rises_long(self) -> None:
        """For a long, if new stop is higher, it must be adopted."""
        sl_engine = DynamicStopLoss(atr_multiplier=2.0)
        current_sl = StopLossLevel(
            stop_price=48_000.0, distance_atr=2.0,
            anchor_type="atr_multiplier", rationale="initial"
        )
        updated_sl = sl_engine.trail_stop(
            current_stop=current_sl,
            direction=1,
            current_price=55_000.0,   # price rose significantly
            current_atr=1_000.0,
            regime_label="bull_calm"
        )
        assert updated_sl.stop_price > current_sl.stop_price

    def test_volatile_regime_widens_sl(self) -> None:
        """'bear_volatile' regime must produce a wider SL (lower for long)."""
        sl_engine = DynamicStopLoss(atr_multiplier=2.0)
        sl_calm = sl_engine.initial_stop(50_000, 1, 1_000, regime_label="bull_calm")
        sl_volatile = sl_engine.initial_stop(50_000, 1, 1_000, regime_label="bear_volatile")
        assert sl_volatile.stop_price < sl_calm.stop_price


# ─────────────────────────────────────────────────────────────────────────────
# 4. Backtest metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestMetrics:

    def _flat_returns(self) -> pd.Series:
        return pd.Series(np.zeros(252))

    def _positive_returns(self) -> pd.Series:
        rng = np.random.default_rng(0)
        return pd.Series(rng.normal(0.001, 0.01, 252))

    def test_sharpe_flat_is_zero(self) -> None:
        assert sharpe_ratio(self._flat_returns()) == pytest.approx(0.0, abs=1e-9)

    def test_sharpe_positive_is_positive(self) -> None:
        assert sharpe_ratio(self._positive_returns()) > 0

    def test_sortino_positive_returns(self) -> None:
        rets = self._positive_returns()
        assert sortino_ratio(rets) >= sharpe_ratio(rets)

    def test_max_drawdown_always_negative_or_zero(self) -> None:
        equity = pd.Series([100.0, 105.0, 98.0, 110.0, 107.0])
        assert max_drawdown(equity) <= 0

    def test_max_drawdown_magnitude(self) -> None:
        equity = pd.Series([100.0, 120.0, 90.0])
        assert max_drawdown(equity) == pytest.approx(-0.25, rel=1e-4)

    def test_profit_factor_all_wins(self) -> None:
        rets = pd.Series([0.01, 0.02, 0.03])
        assert profit_factor(rets) == float("inf")

    def test_profit_factor_all_losses(self) -> None:
        rets = pd.Series([-0.01, -0.02])
        assert profit_factor(rets) == pytest.approx(0.0, abs=1e-9)

    def test_sharpe_uses_365_periods(self) -> None:
        """Default periods_per_year changed to 365 for 24/7 crypto markets."""
        rets = pd.Series([0.001] * 365)
        sr_365 = sharpe_ratio(rets, periods_per_year=365)
        sr_252 = sharpe_ratio(rets, periods_per_year=252)
        # 365 annualisation produces higher Sharpe than 252 for positive returns
        assert sr_365 > sr_252

    def test_regime_breakdown_returns_dataframe(self) -> None:
        """regime_performance_breakdown returns a non-empty DataFrame."""
        from backtest.engine import regime_performance_breakdown
        rng = np.random.default_rng(7)
        rets = pd.Series(rng.normal(0, 0.01, 100))
        regimes = pd.Series(["bull_calm"] * 50 + ["bear_volatile"] * 50, index=rets.index)
        result = regime_performance_breakdown(rets, regimes)
        assert "bull_calm" in result.index
        assert "bear_volatile" in result.index
        assert "sharpe" in result.columns
