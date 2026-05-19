"""
/models/patterns.py — Recurrent pattern detection.

Modules:
1. SeasonalityAnalyser  — intraday / intraweek return seasonality.
2. FourierCycleExtractor — dominant frequency cycles via FFT.
3. CandleClusterer       — Japanese candlestick pattern clustering (k-means
                           on normalised bar metrics).

Design decisions:
- Fourier analysis operates on de-trended log-returns (HP filter or
  first-difference). Raw prices contain a unit root that would dominate
  the spectrum; returns are approximately stationary.
- Seasonality is computed as mean log-return by (hour, weekday) bucket,
  with a t-test to flag statistically significant periods. We filter for
  p < 0.05 to avoid spurious patterns.
- Candlestick clustering: each bar is represented as a 5-element normalised
  vector (body_pct, upper_wick_pct, lower_wick_pct, vol_rel, spread_norm).
  KMeans on this space groups bars into archetypes (e.g. doji, marubozu,
  hammer). The cluster centroid database is used downstream to detect
  high-signal bar formations without hard-coding individual pattern rules.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Seasonality
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalityAnalyser:
    """
    Compute mean log-return by temporal bucket with statistical significance.

    Supports:
    - Intraday: bucket by hour-of-day (requires sub-daily data).
    - Intraweek: bucket by day-of-week (daily data sufficient).
    - Monthly: bucket by month.
    """

    def by_hour(self, df: pd.DataFrame, p_threshold: float = 0.05) -> pd.DataFrame:
        """Return mean log-ret by hour with p-value filter."""
        return self._bucket_analysis(df, df.index.hour, "hour", p_threshold)  # type: ignore[attr-defined]

    def by_weekday(self, df: pd.DataFrame, p_threshold: float = 0.05) -> pd.DataFrame:
        """Return mean log-ret by weekday (0=Mon, 6=Sun)."""
        return self._bucket_analysis(df, df.index.dayofweek, "weekday", p_threshold)  # type: ignore[attr-defined]

    def by_month(self, df: pd.DataFrame, p_threshold: float = 0.05) -> pd.DataFrame:
        """Return mean log-ret by calendar month."""
        return self._bucket_analysis(df, df.index.month, "month", p_threshold)  # type: ignore[attr-defined]

    def _bucket_analysis(
        self,
        df: pd.DataFrame,
        bucket: pd.Series | np.ndarray,
        name: str,
        p_threshold: float,
    ) -> pd.DataFrame:
        """
        Group log_ret by bucket; run one-sample t-test against zero mean.

        Returns DataFrame with [mean_ret, std_ret, t_stat, p_value, significant].
        """
        if "log_ret" not in df.columns:
            raise ValueError("DataFrame must have 'log_ret' column.")

        df_tmp = df[["log_ret"]].copy()
        df_tmp[name] = bucket

        rows = []
        for b, grp in df_tmp.groupby(name):
            returns = grp["log_ret"].dropna().values
            if len(returns) < 10:  # too few observations for reliable t-test
                continue
            t_stat, p_val = stats.ttest_1samp(returns, popmean=0.0)
            rows.append({
                name: b,
                "mean_ret": returns.mean(),
                "std_ret": returns.std(),
                "n": len(returns),
                "t_stat": t_stat,
                "p_value": p_val,
                "significant": p_val < p_threshold,
            })

        result = pd.DataFrame(rows).set_index(name)
        n_sig = result["significant"].sum()
        logger.info("Seasonality by %s: %d/%d buckets significant (p<%.2f)", name, n_sig, len(result), p_threshold)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fourier Cycle Extractor
# ─────────────────────────────────────────────────────────────────────────────

class FourierCycleExtractor:
    """
    Identify dominant periodicity in log-returns using Fast Fourier Transform.

    The spectrum of a financial return series contains many frequencies; we
    extract the top-k by power and return their period (in bars) and amplitude.

    These dominant cycles feed into the timing module to filter entries:
    only enter when price is near a cycle trough (for longs) or crest (shorts).
    """

    def extract_dominant_cycles(
        self,
        log_returns: pd.Series,
        top_k: int = 5,
        min_period_bars: int = 5,
    ) -> pd.DataFrame:
        """
        Args:
            log_returns:    Series of log returns (stationary; do NOT pass prices).
            top_k:          Number of dominant cycles to return.
            min_period_bars: Ignore sub-noise cycles shorter than this.

        Returns:
            DataFrame [period_bars, frequency, amplitude, phase_rad] sorted
            by amplitude descending.
        """
        series = log_returns.dropna().values
        N = len(series)

        # Zero-mean and Hann-window to reduce spectral leakage
        series = series - series.mean()
        window = np.hanning(N)
        spectrum = np.fft.rfft(series * window)
        freqs = np.fft.rfftfreq(N)  # cycles per bar

        amplitudes = np.abs(spectrum)
        phases = np.angle(spectrum)

        # Filter out DC (freq=0) and sub-noise periods
        valid = (freqs > 0) & (1.0 / (freqs + 1e-12) >= min_period_bars)
        filtered_amp = np.where(valid, amplitudes, 0.0)

        top_indices = np.argpartition(filtered_amp, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(filtered_amp[top_indices])[::-1]]

        rows = []
        for i in top_indices:
            freq = freqs[i]
            period = 1.0 / freq if freq > 0 else np.inf
            rows.append({
                "period_bars": round(period, 1),
                "frequency": round(float(freq), 6),
                "amplitude": round(float(amplitudes[i]), 6),
                "phase_rad": round(float(phases[i]), 4),
            })

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Candlestick Clusterer
# ─────────────────────────────────────────────────────────────────────────────

class CandleClusterer:
    """
    Unsupervised clustering of candlestick shapes via k-means.

    Each bar is encoded as a 5-d vector in a scale-free space:
      [body_pct, upper_wick_pct, lower_wick_pct, rel_vol, spread_norm]

    Cluster centroids are saved and reused for inference. Cluster IDs
    are interpreted post-hoc by inspecting centroid values (similar to
    how HMM regimes are labelled).
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans: Optional[KMeans] = None
        self._scaler: StandardScaler = StandardScaler()

    def _encode_candles(self, df: pd.DataFrame) -> np.ndarray:
        """Encode each bar as a 5-d normalised vector."""
        spread = df["high"] - df["low"] + 1e-9
        body = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        vol_ma = df["volume"].rolling(20).mean() + 1e-9

        features = pd.DataFrame({
            "body_pct": body / spread,
            "upper_wick_pct": upper_wick / spread,
            "lower_wick_pct": lower_wick / spread,
            "rel_vol": df["volume"] / vol_ma,
            "spread_norm": spread / df["close"],
        }).dropna()
        return features.values, features.index  # type: ignore[return-value]

    def fit(self, df: pd.DataFrame) -> "CandleClusterer":
        """Fit k-means on historical candle encodings."""
        X, _ = self._encode_candles(df)  # type: ignore[misc]
        X_scaled = self._scaler.fit_transform(X)
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self._kmeans.fit(X_scaled)
        logger.info("CandleClusterer fitted: %d clusters, inertia=%.2f",
                    self.n_clusters, self._kmeans.inertia_)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign cluster labels to each bar in df.

        Returns:
            Series of cluster IDs aligned to df.index.
        """
        if self._kmeans is None:
            raise RuntimeError("Call .fit() first.")
        X, idx = self._encode_candles(df)  # type: ignore[misc]
        X_scaled = self._scaler.transform(X)
        labels = self._kmeans.predict(X_scaled)
        return pd.Series(labels, index=idx, name="candle_cluster")

    def centroids_summary(self) -> pd.DataFrame:
        """Return human-readable centroid table for post-hoc interpretation."""
        if self._kmeans is None:
            raise RuntimeError("Call .fit() first.")
        cols = ["body_pct", "upper_wick_pct", "lower_wick_pct", "rel_vol", "spread_norm"]
        # Inverse-transform to original feature space
        centroids = self._scaler.inverse_transform(self._kmeans.cluster_centers_)
        return pd.DataFrame(centroids, columns=cols).round(4)
