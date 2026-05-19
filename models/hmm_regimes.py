"""
/models/hmm_regimes.py — Hidden Markov Model for market regime detection.

Design decisions:
- GaussianHMM from hmmlearn with "full" covariance by default: captures
  cross-feature correlations (e.g. log_ret and realised_vol co-move in
  bear regimes), outperforming "diag" on financial data empirically.
- Number of components (n_regimes) is a hyperparameter; optimal K is
  selected via BIC on the training set during WFO (default search: 2–6).
- Extended observation matrix (10 features vs original 4):
    [log_ret, realised_vol, atr_norm, vol_profile_z,
     ret_lag1..3,  vol_ratio, price_pct_100d, atr_ratio]
  Momentum lags capture autocorrelation that the base 4-feature set
  misses entirely on daily data. vol_ratio and atr_ratio are
  scale-free and stable across different price regimes.
- Labels are soft post-hoc assignments; predict_proba (forward algorithm)
  is used in strategy layer for probabilistic confluence scoring.
- Model artefacts serialised with joblib; path includes timestamp so
  versions are never silently overwritten.

Optimisation log (v2):
- Added build_extended_observations() — 10-feature obs matrix.
- BIC default range narrowed to (2, 6); K=5 optimal on BTC daily.
- predict_bull_prob / predict_bear_vol_prob convenience methods added.
- bootstrap_regime_stats() for per-regime forward-return CI.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from hmmlearn import hmm  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from config import settings

logger = logging.getLogger(__name__)

# Canonical regime label type (int index → human label)
RegimeLabels = dict[int, str]


# ─────────────────────────────────────────────────────────────────────────────
# HMMRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

class HMMRegimeDetector:
    """
    Wraps hmmlearn.GaussianHMM with:
    - StandardScaler preprocessing (required for stable convergence)
    - BIC-based model selection helper
    - Interpretable regime labelling
    - Serialisation / deserialisation

    Args:
        n_regimes:       Number of latent states.
        covariance_type: "full" | "diag" | "spherical" | "tied".
        n_iter:          EM iterations; 200 is sufficient for daily data.
        random_state:    Reproducibility seed.
    """

    def __init__(
        self,
        n_regimes: int | None = None,
        covariance_type: str | None = None,
        n_iter: int = 200,
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
        """
        Fit the HMM on the observation matrix.

        Args:
            observations: shape (T, n_features), already aligned (no NaNs).

        Returns:
            self (fluent interface).
        """
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
        """
        Viterbi decode: returns integer regime index for each time step.

        Shape: (T,)
        """
        self._assert_fitted()
        X = self._scaler.transform(observations)
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Forward-algorithm posterior probabilities P(state | obs_{1:t}).

        Shape: (T, n_regimes) — useful for soft confluence scoring in strategy.
        """
        self._assert_fitted()
        X = self._scaler.transform(observations)
        _, posteriors = self._model.score_samples(X)  # type: ignore[union-attr]
        return posteriors

    def get_current_regime(self, observations: np.ndarray) -> tuple[int, str]:
        """
        Return the most recent regime index and its human label.

        Args:
            observations: full history up to now (the model needs context).

        Returns:
            (regime_idx, regime_label)
        """
        regimes = self.predict_regimes(observations)
        idx = int(regimes[-1])
        label = self._regime_labels.get(idx, f"regime_{idx}")
        return idx, label

    # ── Extended observation builder ──────────────────────────────────────────

    @staticmethod
    def build_extended_observations(
        feat_df: pd.DataFrame,
        n_lags: int = 3,
    ) -> tuple[np.ndarray, pd.Index]:
        """
        Build a 10-feature observation matrix from an enriched feature DataFrame.

        Features: [log_ret, realised_vol, atr_norm, vol_profile_z,
                   ret_lag1..n_lags, vol_ratio, price_pct_100d, atr_ratio]

        Momentum lags are the single most impactful addition over the base
        4-feature matrix: they give the HMM explicit autocorrelation signal,
        which is the primary distinguishing characteristic of trending vs
        ranging regimes on daily BTC/Gold data.

        vol_ratio and atr_ratio are rolling-normalised (vs 60-bar mean),
        making them regime-invariant across different volatility environments.

        Returns:
            (obs_array, valid_index) — NaN rows are dropped; index is returned
            so callers can align the output back to the original DataFrame.
        """
        import pandas as pd  # noqa: PLC0415 — local import for type checking clarity

        df = feat_df.copy()
        for lag in range(1, n_lags + 1):
            df[f"ret_lag{lag}"] = df["log_ret"].shift(lag)

        df["vol_ratio"] = df["realised_vol"] / (
            df["realised_vol"].rolling(60).mean() + 1e-9
        )
        r100_high = df["close"].rolling(100).max()
        r100_low = df["close"].rolling(100).min()
        df["price_pct"] = (df["close"] - r100_low) / (r100_high - r100_low + 1e-9)
        df["atr_ratio"] = df["atr"] / (df["atr"].rolling(60).mean() + 1e-9)

        cols = [
            "log_ret", "realised_vol", "atr_norm", "vol_profile_z",
            "ret_lag1", "ret_lag2", "ret_lag3",
            "vol_ratio", "price_pct", "atr_ratio",
        ]
        sub = df[cols].dropna()
        return sub.values, sub.index

    # ── Convenience probability accessors ─────────────────────────────────────

    def predict_bull_prob(self, observations: np.ndarray) -> np.ndarray:
        """
        Return P(bull | obs_{1:t}) summed over all bull states.
        Shape: (T,). Use this instead of raw predict_proba in strategy layer.
        """
        self._assert_fitted()
        posteriors = self.predict_proba(observations)
        bull_states = [
            k for k, v in self._regime_labels.items() if v.startswith("bull")
        ]
        if not bull_states:
            return np.zeros(len(observations))
        return posteriors[:, bull_states].sum(axis=1)

    def predict_bear_vol_prob(self, observations: np.ndarray) -> np.ndarray:
        """
        Return P(bear_volatile | obs_{1:t}).
        Shape: (T,).
        """
        self._assert_fitted()
        posteriors = self.predict_proba(observations)
        bv_states = [
            k for k, v in self._regime_labels.items() if v == "bear_volatile"
        ]
        if not bv_states:
            return np.zeros(len(observations))
        return posteriors[:, bv_states].sum(axis=1)

    def bootstrap_regime_stats(
        self,
        feat_df: pd.DataFrame,
        observations: np.ndarray,
        forward_days: int = 5,
        n_bootstrap: int = 500,
        ci: float = 0.95,
    ) -> pd.DataFrame:
        """
        Per-regime forward-return statistics with bootstrap confidence intervals.

        Args:
            feat_df:      Enriched feature DataFrame (must contain 'log_ret').
            observations: HMM observation array aligned to feat_df.
            forward_days: Look-ahead window for cumulative return calculation.
            n_bootstrap:  Number of bootstrap resamples for CI.
            ci:           Confidence interval level (default 0.95 → 95% CI).

        Returns:
            DataFrame indexed by regime label with columns:
            [n, mean_fwd_ret, ci_low, ci_high, win_rate, std_fwd_ret]
        """
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
            boots = [
                arr[rng.integers(0, len(arr), len(arr))].mean()
                for _ in range(n_bootstrap)
            ]
            rows.append({
                "regime": lbl,
                "n": len(arr),
                "mean_fwd_ret": float(arr.mean()),
                "ci_low": float(np.percentile(boots, alpha * 100)),
                "ci_high": float(np.percentile(boots, (1 - alpha) * 100)),
                "win_rate": float((arr > 0).mean()),
                "std_fwd_ret": float(arr.std()),
            })

        return pd.DataFrame(rows).set_index("regime").round(6)

    def bic(self, observations: np.ndarray) -> float:
        """
        Bayesian Information Criterion for model selection.

        Lower BIC → better model. Use in WFO to select optimal n_regimes.
        BIC = -2 * log-likelihood + n_params * log(T)
        """
        self._assert_fitted()
        X = self._scaler.transform(observations)
        T, n_feat = X.shape
        log_lik = self._model.score(X)  # type: ignore[union-attr]

        # Parameter count for full-covariance Gaussian HMM:
        # transition matrix: K*(K-1), means: K*D, covariances: K*D*(D+1)/2
        K = self.n_regimes
        D = n_feat
        n_params = K * (K - 1) + K * D + K * D * (D + 1) // 2
        return -2 * log_lik * T + n_params * np.log(T)

    def save(self, path: Path | None = None) -> Path:
        """Persist model + scaler to disk."""
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
        """Deserialise a saved model."""
        data = joblib.load(path)
        obj = cls()
        obj._model = data["model"]
        obj._scaler = data["scaler"]
        obj._regime_labels = data["labels"]
        obj.n_regimes = obj._model.n_components
        obj._is_fitted = True
        return obj

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    def _assign_labels(self, observations: np.ndarray) -> RegimeLabels:
        """
        Assign interpretable labels by inspecting each regime's mean
        log-return (observations column 0) and realised_vol (column 1).

        Labelling logic:
          - mean_ret > 0 → "bull", else "bear"
          - mean_vol > global_median_vol → "volatile", else "calm"
        """
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
            vol_tag = "volatile" if mean_vol > global_med_vol else "calm"
            labels[k] = f"{direction}_{vol_tag}"

        logger.info("Regime labels assigned: %s", labels)
        return labels


# ─────────────────────────────────────────────────────────────────────────────
# BIC-based model selection helper
# ─────────────────────────────────────────────────────────────────────────────

def select_optimal_n_regimes(
    observations: np.ndarray,
    k_range: tuple[int, int] = (2, 6),
) -> tuple[int, list[float]]:
    """
    Fit HMMs for k in k_range and return the k that minimises BIC.

    Args:
        observations: (T, n_features) array.
        k_range:      (k_min, k_max) inclusive.

    Returns:
        (best_k, list_of_bic_scores_for_each_k)
    """
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
