"""HMM + DQN pipeline — traditional two-stage baseline.

Stage 1: Fit a Gaussian HMM on the return series to infer regimes.
Stage 2: Append the HMM-inferred regime (as one-hot) to observations
         and train a standard DQN.

This represents the classical "detect-then-trade" approach and serves
as a strong baseline that explicitly models regime switching but does
not learn the regime detector end-to-end with the trading policy.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture  # lightweight HMM proxy

from .config import REGIME_LABELS
from .dqn import DQNAgent


# ---------------------------------------------------------------------------
# Lightweight HMM-like regime detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Gaussian Mixture Model regime detector.

    We use a GMM as a lightweight proxy for a full HMM because
    scikit-learn's hmmlearn dependency is optional and GMM captures the
    core idea: cluster return windows into latent states.
    """

    def __init__(self, n_components: int = 4, window_size: int = 10, seed: int = 42):
        self.n_components = n_components
        self.window_size = window_size
        self.seed = seed
        self.model: GaussianMixture | None = None

    def fit(self, returns: np.ndarray) -> None:
        """Fit on rolling-window features of the return series."""
        features = self._build_features(returns)
        if len(features) < self.n_components:
            return
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            n_init=3,
            random_state=self.seed,
        )
        self.model.fit(features)

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Return component labels for each time step (after warm-up)."""
        features = self._build_features(returns)
        if self.model is None or len(features) == 0:
            return np.zeros(len(features), dtype=np.int64)
        return self.model.predict(features).astype(np.int64)

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Return soft component probabilities: (T - window + 1, n_components)."""
        features = self._build_features(returns)
        if self.model is None or len(features) == 0:
            return np.zeros((len(features), self.n_components), dtype=np.float64)
        return self.model.predict_proba(features)

    def infer_single(self, recent_returns: np.ndarray) -> np.ndarray:
        """Infer regime probabilities for the most recent window.

        Returns a (n_components,) probability vector.
        """
        if self.model is None:
            return np.ones(self.n_components, dtype=np.float64) / self.n_components
        feat = self._single_feature(recent_returns)
        return self.model.predict_proba(feat.reshape(1, -1)).squeeze(0)

    def _build_features(self, returns: np.ndarray) -> np.ndarray:
        """Rolling-window feature matrix: [mean, std, skew-proxy]."""
        r = np.asarray(returns, dtype=np.float64)
        n = len(r)
        w = self.window_size
        if n < w:
            return np.empty((0, 3), dtype=np.float64)
        rows: list[np.ndarray] = []
        for i in range(w, n + 1):
            window = r[i - w : i]
            rows.append(self._single_feature(window))
        return np.array(rows, dtype=np.float64)

    @staticmethod
    def _single_feature(window: np.ndarray) -> np.ndarray:
        mean = float(np.mean(window))
        std = float(np.std(window)) + 1e-10
        skew = float(np.mean(((window - mean) / std) ** 3))
        return np.array([mean, std, skew], dtype=np.float64)


# ---------------------------------------------------------------------------
# HMM-augmented DQN agent
# ---------------------------------------------------------------------------

class HMMDQNAgent(DQNAgent):
    """DQN that uses a pre-fitted GMM regime detector.

    During training the detector is fitted on the warm-up returns of each
    episode, then the inferred regime probabilities are appended to the
    observation vector at each step.
    """

    def __init__(
        self,
        base_observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        learning_rate: float,
        gamma: float,
        tau: float,
        replay_capacity: int,
        batch_size: int,
        device: str,
        seed: int,
        n_components: int = 4,
        window_size: int = 10,
    ):
        # observation = base features + GMM probabilities
        augmented_dim = base_observation_dim + n_components
        super().__init__(
            observation_dim=augmented_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        self._base_dim = base_observation_dim
        self._n_components = n_components
        self.detector = RegimeDetector(n_components=n_components, window_size=window_size, seed=seed)

    def fit_detector(self, returns: np.ndarray) -> None:
        """Fit / refit the regime detector on a return series."""
        self.detector.fit(returns)

    def augment_state(self, base_state: np.ndarray, recent_returns: np.ndarray) -> np.ndarray:
        """Append GMM regime probabilities to the base observation."""
        probs = self.detector.infer_single(recent_returns).astype(np.float32)
        return np.concatenate([np.asarray(base_state, dtype=np.float32), probs])
