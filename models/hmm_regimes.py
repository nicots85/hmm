"""
/models/hmm_regimes.py — Hidden Markov Model for market regime detection.

Design decisions:
- GaussianHMM from hmmlearn with "full" covariance by default: captures
  cross-feature correlations (e.g. log_ret and realised_vol co-move in
  bear regimes), outperforming "diag" on financial data empirically.
- Number of components (n_regimes) is a hyperparameter, defaulting to 4:
  {low-vol bull, high-vol bull, low-vol bear, high-vol/crisis}.
  Optimal K is selected via BIC on the training set during WFO.
- We label regimes post-hoc by their mean log-return and mean realised_vol,
  giving interpretable labels: "bull_calm", "bull_volatile", "bear_calm",
  "bear_volatile". Labels are soft — they describe the regime's average
  character, not a guarantee.
- Model artefacts are serialised with joblib (faster than pickle for numpy
  arrays). The serialisation path includes a timestamp so versions are
  never silently overwritten.
- The Viterbi path (most-likely hidden state sequence) is used for regime
  assignment; predict_proba gives forward-looking soft probabilities for
  use in the strategy confluence layer.
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
    k_range: tuple[int, int] = (2, 8),
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
