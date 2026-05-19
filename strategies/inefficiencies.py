"""
/strategies/inefficiencies.py — Market inefficiency detectors.

Three inefficiency types, each with an integrated historical win-rate validator:

1. BTCWeekendGap
   Rationale: BTC perpetual futures trade 24/7; traditional markets close
   Friday. CME Bitcoin futures settle weekly. The gap between Friday 17:00 CT
   close and Sunday re-open has historically shown mean-reversion bias.
   Implementation: detect gap > threshold * ATR, classify as fill-up / fill-down,
   record forward return at T+4h, T+8h, T+24h.

2. LondonGoldFix
   Rationale: LBMA AM fix (10:30 London) and PM fix (15:00 London) coincide
   with institutionally-driven order flow that creates statistically
   detectable micro-trends in XAUUSD.
   Implementation: check if current bar falls in the ±15-min window of each fix;
   record direction and forward return.

3. TemporalArbitrageFilter
   Rationale: Validates any time-based signal against its walk-forward
   win-rate. A signal is only marked "active" if the historical win rate
   strictly exceeds the configured threshold (default: 60%).
   Implementation: rolling win-rate over the out-of-sample window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")
LONDON_TZ = ZoneInfo("Europe/London")
CT_TZ = ZoneInfo("America/Chicago")


# ─────────────────────────────────────────────────────────────────────────────
# 1. BTC Weekend Gap
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeekendGapConfig:
    gap_atr_multiplier: float = 0.5     # min gap size relative to ATR
    forward_bars: list[int] = field(default_factory=lambda: [4, 8, 24])
    timeframe_hours: int = 1            # assumed bar duration in hours


class BTCWeekendGap:
    """
    Detects BTC weekend price gaps and measures fill probability.

    A 'gap' is defined as |open_sunday - close_friday| > multiplier × ATR.
    Direction convention:
      +1 = gap_up  (open > close_fri) → fill-down bias (short signal)
      -1 = gap_down (open < close_fri) → fill-up bias (long signal)
    """

    def __init__(self, cfg: WeekendGapConfig | None = None) -> None:
        self.cfg = cfg or WeekendGapConfig()

    def detect_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify weekend gaps in a BTC OHLCV DataFrame.

        Args:
            df: UTC-indexed hourly (or coarser) OHLCV with 'atr' column.

        Returns:
            Subset DataFrame of gap events with columns:
            [gap_size, gap_direction, close_friday, open_sunday, atr_at_gap]
        """
        if "atr" not in df.columns:
            raise ValueError("'atr' column required — run FeatureEngineer first.")

        # Identify Friday closes (last bar of Friday UTC)
        is_friday = df.index.dayofweek == 4  # type: ignore[attr-defined]
        is_sunday = df.index.dayofweek == 6  # type: ignore[attr-defined]

        friday_closes = df[is_friday]["close"]
        sunday_opens = df[is_sunday]["open"]

        gaps: list[dict] = []
        for sun_ts, sun_open in sunday_opens.items():
            # Find the nearest preceding Friday close
            fri_closes_before = friday_closes[friday_closes.index < sun_ts]  # type: ignore[operator]
            if fri_closes_before.empty:
                continue
            fri_ts = fri_closes_before.index[-1]
            fri_close = fri_closes_before.iloc[-1]
            atr_val = df.loc[fri_ts, "atr"]

            gap_size = sun_open - fri_close
            if abs(gap_size) < self.cfg.gap_atr_multiplier * atr_val:
                continue  # below significance threshold

            gaps.append({
                "timestamp": sun_ts,
                "gap_size": gap_size,
                "gap_direction": np.sign(gap_size),   # +1 up, -1 down
                "close_friday": fri_close,
                "open_sunday": sun_open,
                "atr_at_gap": atr_val,
            })

        if not gaps:
            logger.info("No significant weekend gaps found.")
            return pd.DataFrame()

        result = pd.DataFrame(gaps).set_index("timestamp")
        logger.info("Detected %d weekend gaps.", len(result))
        return result

    def compute_fill_statistics(
        self, df: pd.DataFrame, gaps: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each gap event, measure whether price returned to the pre-gap close
        within each forward_bars window. Returns fill_rate per horizon.
        """
        records = []
        for ts, row in gaps.iterrows():
            target_close = row["close_friday"]
            direction = row["gap_direction"]

            for fwd in self.cfg.forward_bars:
                try:
                    future = df.loc[ts:].iloc[1: fwd + 1]["close"]  # type: ignore[misc]
                except KeyError:
                    continue
                if future.empty:
                    continue

                # Fill = price crossed the gap-fill level
                if direction > 0:  # gap up → fill means price came down to close_fri
                    filled = (future <= target_close).any()
                else:              # gap down → fill means price rose to close_fri
                    filled = (future >= target_close).any()

                records.append({"horizon_bars": fwd, "filled": int(filled)})

        stats_df = (
            pd.DataFrame(records)
            .groupby("horizon_bars")["filled"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "fill_rate", "count": "n_events"})
        )
        logger.info("Gap fill statistics:\n%s", stats_df.to_string())
        return stats_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. London Gold Fix Windows
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldFixConfig:
    am_fix_hour_london: int = 10
    am_fix_minute_london: int = 30
    pm_fix_hour_london: int = 15
    pm_fix_minute_london: int = 0
    window_minutes: int = 15    # ±minutes around fix for signal window


class LondonGoldFixDetector:
    """
    Detects LBMA AM/PM fix windows and computes directional bias.

    Signal logic:
    - AM fix (10:30 London): morning price-setting. Historical bias shows
      mean-reversion from spikes into the window.
    - PM fix (15:00 London): stronger institutional flow; tendency to
      trend-follow in the direction set during the window.
    """

    def __init__(self, cfg: GoldFixConfig | None = None) -> None:
        self.cfg = cfg or GoldFixConfig()

    def flag_fix_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add boolean columns [in_am_fix, in_pm_fix] to the DataFrame.
        Requires sub-daily (≤1h) UTC-indexed data.
        """
        df = df.copy()
        london_index = df.index.tz_convert(LONDON_TZ)  # type: ignore[attr-defined]

        def _in_window(idx: pd.DatetimeIndex, fix_hour: int, fix_min: int, w: int) -> pd.Series:
            fix_minutes = fix_hour * 60 + fix_min
            bar_minutes = idx.hour * 60 + idx.minute
            return pd.Series(
                (bar_minutes >= fix_minutes - w) & (bar_minutes <= fix_minutes + w),
                index=df.index,
            )

        df["in_am_fix"] = _in_window(
            london_index, self.cfg.am_fix_hour_london,
            self.cfg.am_fix_minute_london, self.cfg.window_minutes
        )
        df["in_pm_fix"] = _in_window(
            london_index, self.cfg.pm_fix_hour_london,
            self.cfg.pm_fix_minute_london, self.cfg.window_minutes
        )
        return df

    def compute_fix_statistics(
        self, df: pd.DataFrame, forward_bars: int = 4
    ) -> dict[str, float]:
        """
        Compute directional win-rates for bars inside AM and PM fix windows.

        Returns:
            Dict with keys [am_long_wr, am_short_wr, pm_long_wr, pm_short_wr].
        """
        if "log_ret" not in df.columns:
            raise ValueError("Requires 'log_ret' — run FeatureEngineer first.")

        result: dict[str, float] = {}
        for fix_label, col in [("am", "in_am_fix"), ("pm", "in_pm_fix")]:
            if col not in df.columns:
                df = self.flag_fix_windows(df)
            in_fix = df[df[col]]
            if in_fix.empty:
                result[f"{fix_label}_long_wr"] = 0.5
                result[f"{fix_label}_short_wr"] = 0.5
                continue

            fwd_rets = [
                df["log_ret"].iloc[
                    df.index.get_loc(ts) + 1:  # type: ignore[misc]
                    df.index.get_loc(ts) + forward_bars + 1
                ].sum()
                for ts in in_fix.index
                if df.index.get_loc(ts) + forward_bars < len(df)  # type: ignore[misc]
            ]
            if not fwd_rets:
                result[f"{fix_label}_long_wr"] = 0.5
                result[f"{fix_label}_short_wr"] = 0.5
                continue

            fwd_arr = np.array(fwd_rets)
            long_wr = float((fwd_arr > 0).mean())
            result[f"{fix_label}_long_wr"] = long_wr
            result[f"{fix_label}_short_wr"] = 1.0 - long_wr

        logger.info("Gold fix win-rates: %s", result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Walk-Forward Win-Rate Validator (gate for all inefficiency signals)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalArbitrageFilter:
    """
    Rolling walk-forward win-rate gate.

    Any time-based signal must pass this filter before reaching execution.
    The filter computes the win-rate over the last `lookback_bars` bars
    and returns True only if wr > min_win_rate.

    This is the "Filtro de Ineficiencias" required by the architecture spec:
    blocks signals if historical win-rate ≤ 60%.
    """

    def __init__(
        self,
        min_win_rate: float = 0.60,
        lookback_bars: int = 100,
    ) -> None:
        self.min_win_rate = min_win_rate
        self.lookback_bars = lookback_bars

    def validate(
        self,
        signal_mask: pd.Series,
        forward_returns: pd.Series,
    ) -> tuple[bool, float]:
        """
        Check if a signal's rolling win-rate exceeds the threshold.

        Args:
            signal_mask:     Boolean Series — True where signal fired.
            forward_returns: Scalar returns achieved after each signal.

        Returns:
            (is_valid, rolling_win_rate)
        """
        signal_returns = forward_returns[signal_mask].dropna()
        recent = signal_returns.iloc[-self.lookback_bars:]

        if len(recent) < 10:
            logger.warning("Insufficient signal history (%d bars). Signal blocked.", len(recent))
            return False, 0.0

        wr = float((recent > 0).mean())
        is_valid = wr > self.min_win_rate
        if not is_valid:
            logger.info(
                "Signal blocked: rolling WR=%.2f%% < %.0f%% threshold.",
                wr * 100, self.min_win_rate * 100
            )
        return is_valid, wr
