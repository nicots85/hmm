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
    """Minimal OHLCV DataFrame for feature engineering tests."""
    n = 200
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
        # First row is NaN from shift; transform() calls dropna()
        assert not out["log_ret"].isna().any()

    def test_log_returns_numerical_accuracy(self) -> None:
        """log_ret = ln(c_t / c_{t-1}); verify against manual calculation."""
        prices = pd.Series([100.0, 105.0, 99.75])
        expected = [np.log(105.0 / 100.0), np.log(99.75 / 105.0)]
        index = pd.date_range("2023-01-01", periods=3, freq="1D", tz="UTC")
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.01,
            "low": prices * 0.99, "close": prices, "volume": [1e6] * 3,
        }, index=index)

        fe = FeatureEngineer(FeatureConfig(atr_period=2, realised_vol_window=2,
                                           vol_profile_window=2))
        out = fe.transform(df)
        computed = out["log_ret"].values
        np.testing.assert_allclose(computed, expected, rtol=1e-10)

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        fe = FeatureEngineer()
        out = fe.transform(sample_ohlcv)
        assert (out["atr"] > 0).all()

    def test_hmm_observations_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        fe = FeatureEngineer()
        out = fe.transform(sample_ohlcv)
        obs = fe.get_hmm_observations(out)
        assert obs.ndim == 2
        assert obs.shape[1] == 4    # [log_ret, realised_vol, atr_norm, vol_profile_z]
        assert not np.isnan(obs).any()


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
        assert sortino_ratio(rets) >= sharpe_ratio(rets)  # Sortino ≥ Sharpe for positive drift

    def test_max_drawdown_always_negative_or_zero(self) -> None:
        equity = pd.Series([100.0, 105.0, 98.0, 110.0, 107.0])
        dd = max_drawdown(equity)
        assert dd <= 0

    def test_max_drawdown_magnitude(self) -> None:
        equity = pd.Series([100.0, 120.0, 90.0])
        # peak=120, trough=90 → dd = (90-120)/120 = -0.25
        assert max_drawdown(equity) == pytest.approx(-0.25, rel=1e-4)

    def test_profit_factor_all_wins(self) -> None:
        rets = pd.Series([0.01, 0.02, 0.03])
        assert profit_factor(rets) == float("inf")

    def test_profit_factor_all_losses(self) -> None:
        rets = pd.Series([-0.01, -0.02])
        assert profit_factor(rets) == pytest.approx(0.0, abs=1e-9)
