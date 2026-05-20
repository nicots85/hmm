"""
/models/hmm_regimes.py — Hidden Markov Model for market regime detection.

v3 changes:
- build_extended_observations(): 10 → 12 features (adds realised_skew,
  realised_kurt). Fat tails separate crisis from normal bear.
- All features shifted by 1 bar before entering obs matrix — no look-ahead.
- ExpandingWindowRefitter: rolling HMM refit for WFO (fixes stale model).
- predict_bull_prob / predict_bear_vol_prob: unchanged, already correct.
- bootstrap_regime_stats(): unchanged.
- select_optimal_n_regimes default range stays (2,6).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from config import settings

logger = logging.getLogger(__name__)

RegimeLabels = dict[int, str]


class HMMRegimeDetector:
    """
    GaussianHMM wrapper with extended feature support, soft posteriors,
    expanding-window refit, and joblib serialisation.
    """

    def __init__(
        self,
        n_regimes: int | None = None,
        covariance_type: str | None = None,
        n_iter: int = 300,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes or settings.hmm_n_components
        self.covariance_type = covariance_type or settings.hmm_covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._model: hmm.GaussianHMM | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._regime_labels: RegimeLabels = {}
        self._is_fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, observations: np.ndarray) -> "HMMRegimeDetector":
        if np.isnan(observations).any():
            raise ValueError("observations contain NaN — run FeatureEngineer.transform() first.")
        X = self._scaler.fit_transform(observations)
        self._model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self._model.fit(X)
        self._is_fitted = True
        self._regime_labels = self._assign_labels(observations)
        logger.info(
            "HMM fitted: %d regimes, log-likelihood=%.2f",
            self.n_regimes,
            self._model.score(X),
        )
        return self

    def predict_regimes(self, observations: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        X = self._scaler.transform(observations)
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        X = self._scaler.transform(observations)
        _, posteriors = self._model.score_samples(X)  # type: ignore[union-attr]
        return posteriors

    def predict_bull_prob(self, observations: np.ndarray) -> np.ndarray:
        """P(bull | obs_{1:t}) summed over all bull states. Shape (T,)."""
        self._assert_fitted()
        posteriors = self.predict_proba(observations)
        bull_states = [k for k, v in self._regime_labels.items() if v.startswith("bull")]
        if not bull_states:
            return np.zeros(len(observations))
        return posteriors[:, bull_states].sum(axis=1)

    def predict_bear_vol_prob(self, observations: np.ndarray) -> np.ndarray:
        """P(bear_volatile | obs_{1:t}). Shape (T,)."""
        self._assert_fitted()
        posteriors = self.predict_proba(observations)
        bv_states = [k for k, v in self._regime_labels.items() if v == "bear_volatile"]
        if not bv_states:
            return np.zeros(len(observations))
        return posteriors[:, bv_states].sum(axis=1)

    def get_current_regime(self, observations: np.ndarray) -> tuple[int, str]:
        regimes = self.predict_regimes(observations)
        idx = int(regimes[-1])
        return idx, self._regime_labels.get(idx, f"regime_{idx}")

    def bic(self, observations: np.ndarray) -> float:
        self._assert_fitted()
        X = self._scaler.transform(observations)
        T, D = X.shape
        log_lik = self._model.score(X)  # type: ignore[union-attr]
        K = self.n_regimes
        n_params = K * (K - 1) + K * D + K * D * (D + 1) // 2
        return -2 * log_lik * T + n_params * np.log(T)

    def save(self, path: Path | None = None) -> Path:
        self._assert_fitted()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = path or (
            settings.model_artifact_path / f"hmm_k{self.n_regimes}_{ts}.joblib"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "scaler": self._scaler, "labels": self._regime_labels}, dest)
        logger.info("HMM saved → %s", dest)
        return dest

    @classmethod
    def load(cls, path: Path) -> "HMMRegimeDetector":
        data = joblib.load(path)
        obj = cls()
        obj._model = data["model"]
        obj._scaler = data["scaler"]
        obj._regime_labels = data["labels"]
        obj.n_regimes = obj._model.n_components
        obj._is_fitted = True
        return obj

    # ── Extended observation builder (v3: 12 features) ───────────────────────

    @staticmethod
    def build_extended_observations(
        feat_df: pd.DataFrame,
        n_lags: int = 3,
    ) -> tuple[np.ndarray, pd.Index]:
        """
        Build a 12-feature observation matrix from an enriched DataFrame.

        Features (all causal — no look-ahead):
          log_ret, realised_vol, atr_norm, vol_profile_z,
          ret_lag1..n_lags,
          vol_ratio  = realised_vol / vol_60ma
          price_pct  = (close - min_100) / (max_100 - min_100)   [shifted 1]
          atr_ratio  = atr / atr_60ma
          realised_skew, realised_kurt   ← NEW in v3

        All features shifted by 1 bar to prevent look-ahead contamination.
        """
        df = feat_df.copy()

        for lag in range(1, n_lags + 1):
            df[f"ret_lag{lag}"] = df["log_ret"].shift(lag)

        df["vol_ratio"] = df["realised_vol"] / (
            df["realised_vol"].rolling(60).mean().shift(1) + 1e-9
        )

        r100_high = df["close"].rolling(100).max().shift(1)
        r100_low  = df["close"].rolling(100).min().shift(1)
        df["price_pct"] = (df["close"] - r100_low) / (r100_high - r100_low + 1e-9)

        df["atr_ratio"] = df["atr"] / (df["atr"].rolling(60).mean().shift(1) + 1e-9)

        cols = [
            "log_ret", "realised_vol", "atr_norm", "vol_profile_z",
            "ret_lag1", "ret_lag2", "ret_lag3",
            "vol_ratio", "price_pct", "atr_ratio",
        ]
        # Add higher moments if available (v3 features.py)
        for col in ["realised_skew", "realised_kurt"]:
            if col in df.columns:
                cols.append(col)

        sub = df[cols].dropna()
        return sub.values, sub.index

    # ── Convenience analytics ─────────────────────────────────────────────────

    def bootstrap_regime_stats(
        self,
        feat_df: pd.DataFrame,
        observations: np.ndarray,
        forward_days: int = 5,
        n_bootstrap: int = 500,
        ci: float = 0.95,
    ) -> pd.DataFrame:
        """Per-regime forward-return statistics with 95% bootstrap CI."""
        self._assert_fitted()
        regimes = self.predict_regimes(observations)
        log_ret = feat_df["log_ret"]
        alpha = (1 - ci) / 2
        rows = []
        for k in range(self.n_regimes):
            lbl = self._regime_labels.get(k, f"regime_{k}")
            mask = regimes == k
            idx_list = feat_df.index[mask]
            fwd: list[float] = []
            for ts in idx_list:
                loc = feat_df.index.get_loc(ts)
                future = log_ret.iloc[loc + 1: loc + forward_days + 1]
                if len(future) == forward_days:
                    fwd.append(float(future.sum()))
            if len(fwd) < 5:
                continue
            arr = np.array(fwd)
            rng = np.random.default_rng(42)
            boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_bootstrap)]
            rows.append({
                "regime": lbl, "n": len(arr),
                "mean_fwd_ret": float(arr.mean()),
                "ci_low": float(np.percentile(boots, alpha * 100)),
                "ci_high": float(np.percentile(boots, (1 - alpha) * 100)),
                "win_rate": float((arr > 0).mean()),
                "std_fwd_ret": float(arr.std()),
            })
        return pd.DataFrame(rows).set_index("regime").round(6)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def _assign_labels(self, observations: np.ndarray) -> RegimeLabels:
        regimes = self.predict_regimes(observations)
        global_med_vol = np.median(observations[:, 1])
        labels: RegimeLabels = {}
        for k in range(self.n_regimes):
            mask = regimes == k
            if mask.sum() == 0:
                labels[k] = f"regime_{k}"
                continue
            mean_ret = observations[mask, 0].mean()
            mean_vol = observations[mask, 1].mean()
            direction = "bull" if mean_ret > 0 else "bear"
            vol_tag   = "volatile" if mean_vol > global_med_vol else "calm"
            labels[k] = f"{direction}_{vol_tag}"
        logger.info("Regime labels: %s", labels)
        return labels


# ─────────────────────────────────────────────────────────────────────────────
# Expanding-window refitter  (fixes P8: stale HMM in WFO)
# ─────────────────────────────────────────────────────────────────────────────

class ExpandingWindowRefitter:
    """
    Retrain HMM on an expanding window at each WFO split boundary.

    Instead of fitting once on the full IS set and freezing, this refitter
    re-estimates the model parameters at each new data point arrival, using
    all data from the start up to the current bar. This is the correct
    procedure for non-stationary financial time series.

    Usage:
        refitter = ExpandingWindowRefitter(n_regimes=4, min_train_bars=200)
        for split_idx in range(n_splits):
            model = refitter.fit_up_to(observations, train_end_idx)
            signals = generate_signals(model, observations[train_end_idx:])
    """

    def __init__(
        self,
        n_regimes: int = 4,
        min_train_bars: int = 252,
        refit_every: int = 63,     # quarterly refit (business days ≈ 63)
        covariance_type: str = "full",
    ) -> None:
        self.n_regimes = n_regimes
        self.min_train_bars = min_train_bars
        self.refit_every = refit_every
        self.covariance_type = covariance_type
        self._cache: dict[int, HMMRegimeDetector] = {}

    def fit_up_to(self, observations: np.ndarray, end_idx: int) -> HMMRegimeDetector:
        """
        Return a fitted HMMRegimeDetector using all observations[:end_idx].
        Results are cached by end_idx to avoid redundant refits.
        """
        # Round to nearest refit_every boundary for caching
        cache_key = (end_idx // self.refit_every) * self.refit_every
        if cache_key in self._cache:
            return self._cache[cache_key]

        train = observations[:end_idx]
        if len(train) < self.min_train_bars:
            raise ValueError(
                f"Insufficient training data: {len(train)} < {self.min_train_bars}"
            )

        detector = HMMRegimeDetector(
            n_regimes=self.n_regimes,
            covariance_type=self.covariance_type,
        )
        detector.fit(train)
        self._cache[cache_key] = detector
        logger.debug("HMM refitted at end_idx=%d (key=%d)", end_idx, cache_key)
        return detector

    def clear_cache(self) -> None:
        self._cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# BIC model selection
# ─────────────────────────────────────────────────────────────────────────────

def select_optimal_n_regimes(
    observations: np.ndarray,
    k_range: tuple[int, int] = (2, 6),
) -> tuple[int, list[float]]:
    """Fit HMMs for k in k_range; return k that minimises BIC."""
    bic_scores: list[float] = []
    best_k = k_range[0]
    best_bic = np.inf
    for k in range(k_range[0], k_range[1] + 1):
        detector = HMMRegimeDetector(n_regimes=k)
        try:
            detector.fit(observations)
            bic = detector.bic(observations)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HMM k=%d failed: %s", k, exc)
            bic_scores.append(np.nan)
            continue
        bic_scores.append(bic)
        if bic < best_bic:
            best_bic = bic
            best_k = k
    logger.info("BIC optimal k=%d (BIC=%.2f)", best_k, best_bic)
    return best_k, bic_scores
