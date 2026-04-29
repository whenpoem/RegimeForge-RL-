"""HMM + DQN pipeline — traditional two-stage baseline.

Stage 1: Fit a Gaussian HMM on the return series to infer regimes.
         Falls back to GMM if hmmlearn is not installed.
Stage 2: Append the HMM-inferred regime (as one-hot) to observations
         and train a standard DQN.

This represents the classical "detect-then-trade" approach and serves
as a strong baseline that explicitly models regime switching but does
not learn the regime detector end-to-end with the trading policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

try:
    from hmmlearn.hmm import GaussianHMM

    _HAS_HMM = True
except ImportError:
    GaussianHMM = None  # type: ignore[assignment,misc]
    _HAS_HMM = False

from .config import REGIME_LABELS
from .dqn import DQNAgent


def hmm_available() -> bool:
    """Return True if hmmlearn is installed and importable."""
    return _HAS_HMM


# ---------------------------------------------------------------------------
# Regime detector (HMM if available, GMM fallback)
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Regime detector using HMM (preferred) or GMM (fallback).

    When hmmlearn is installed, uses a full Hidden Markov Model with
    Gaussian emissions, which captures temporal transition dynamics.
    Otherwise falls back to a static GMM on rolling-window features.
    """

    def __init__(
        self,
        n_components: int = 4,
        window_size: int = 10,
        seed: int = 42,
        use_hmm: bool = True,
        covariance_type: str = "full",
    ):
        self.n_components = n_components
        self.window_size = window_size
        self.seed = seed
        self.use_hmm = use_hmm and _HAS_HMM
        self.covariance_type = covariance_type
        self.model: Any = None  # GaussianHMM | GaussianMixture | None
        self._mode = "hmm" if self.use_hmm else "gmm"

    def fit(self, returns: np.ndarray) -> None:
        """Fit detector on a return series."""
        if self.use_hmm:
            self._fit_hmm(returns)
        else:
            self._fit_gmm(returns)

    def _fit_hmm(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=np.float64).reshape(-1, 1)
        if len(r) < self.n_components * 2:
            return
        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=100,
            random_state=self.seed,
            tol=1e-4,
        )
        try:
            model.fit(r)
            self.model = model
        except Exception:
            # HMM fit can fail on degenerate data; fall back to GMM
            self._fit_gmm(returns)
            self._mode = "gmm_fallback"

    def _fit_gmm(self, returns: np.ndarray) -> None:
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
        """Return component labels for each time step."""
        if self.model is None:
            n = len(returns) - self.window_size + 1 if not self.use_hmm else len(returns)
            return np.zeros(max(n, 0), dtype=np.int64)
        if self._mode.startswith("hmm"):
            r = np.asarray(returns, dtype=np.float64).reshape(-1, 1)
            return self.model.predict(r).astype(np.int64)
        features = self._build_features(returns)
        if len(features) == 0:
            return np.array([], dtype=np.int64)
        return self.model.predict(features).astype(np.int64)

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Return soft component probabilities."""
        if self.model is None:
            n = len(returns) - self.window_size + 1 if not self.use_hmm else len(returns)
            return np.zeros((max(n, 0), self.n_components), dtype=np.float64)
        if self._mode.startswith("hmm"):
            r = np.asarray(returns, dtype=np.float64).reshape(-1, 1)
            return self.model.predict_proba(r)
        features = self._build_features(returns)
        if len(features) == 0:
            return np.empty((0, self.n_components), dtype=np.float64)
        return self.model.predict_proba(features)

    def infer_single(self, recent_returns: np.ndarray) -> np.ndarray:
        """Infer regime probabilities for the most recent window.

        Returns a (n_components,) probability vector.
        """
        if self.model is None:
            return np.ones(self.n_components, dtype=np.float64) / self.n_components
        if self._mode.startswith("hmm"):
            r = np.asarray(recent_returns, dtype=np.float64).reshape(-1, 1)
            return self.model.predict_proba(r)[-1]
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "window_size": self.window_size,
            "seed": self.seed,
            "use_hmm": self.use_hmm,
            "covariance_type": self.covariance_type,
            "mode": self._mode,
            "model": self.model,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.n_components = int(state.get("n_components", self.n_components))
        self.window_size = int(state.get("window_size", self.window_size))
        self.seed = int(state.get("seed", self.seed))
        self.use_hmm = bool(state.get("use_hmm", self.use_hmm))
        self.covariance_type = str(state.get("covariance_type", self.covariance_type))
        self._mode = str(state.get("mode", "gmm"))
        model = state.get("model")
        if model is not None and isinstance(model, (GaussianMixture,)):
            self.model = model
        elif model is not None and _HAS_HMM and isinstance(model, GaussianHMM):
            self.model = model
        else:
            self.model = None


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
        use_hmm: bool = True,
        **kwargs,
    ):
        # observation = base features + regime probabilities
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
            **kwargs,
        )
        self._base_dim = base_observation_dim
        self._n_components = n_components
        self.detector = RegimeDetector(
            n_components=n_components, window_size=window_size, seed=seed, use_hmm=use_hmm,
        )

    def fit_detector(self, returns: np.ndarray) -> None:
        """Fit / refit the regime detector on a return series."""
        self.detector.fit(returns)

    def augment_state(self, base_state: np.ndarray, recent_returns: np.ndarray) -> np.ndarray:
        """Append GMM regime probabilities to the base observation."""
        probs = self.detector.infer_single(recent_returns).astype(np.float32)
        return np.concatenate([np.asarray(base_state, dtype=np.float32), probs])

    def save_checkpoint(self, directory: str | Path) -> None:
        super().save_checkpoint(directory)
        torch.save(self.detector.state_dict(), Path(directory) / "detector.pt")

    def load_checkpoint(self, directory: str | Path, weights_only: bool = True) -> None:
        super().load_checkpoint(directory, weights_only=weights_only)
        detector_path = Path(directory) / "detector.pt"
        if detector_path.exists():
            detector_state = torch.load(detector_path, map_location="cpu", weights_only=False)
            if isinstance(detector_state, dict):
                self.detector.load_state_dict(detector_state)


class HMMContinuousAgent:
    """Continuous agent that augments observations with HMM regime probabilities.

    Wraps ContinuousActorCriticAgent — the only difference is the
    observation dimension is base_obs_dim + n_regimes, and observations
    are augmented with regime probabilities before being passed to the
    underlying agent.
    """

    def __init__(
        self,
        base_observation_dim: int,
        action_dim: int,
        config: Any,
        device: str = "cpu",
        n_components: int = 4,
        window_size: int = 10,
        use_hmm: bool = True,
    ):
        from .continuous_agent import ContinuousActorCriticAgent

        self._base_dim = base_observation_dim
        self._n_components = n_components
        self.detector = RegimeDetector(
            n_components=n_components, window_size=window_size, seed=config.seed, use_hmm=use_hmm,
        )
        augmented_dim = base_observation_dim + n_components
        self.agent = ContinuousActorCriticAgent(
            config,
            observation_dim=augmented_dim,
            action_dim=action_dim,
            device=device,
        )

    def fit_detector(self, returns: np.ndarray) -> None:
        self.detector.fit(returns)

    def augment_state(self, base_state: np.ndarray, recent_returns: np.ndarray) -> np.ndarray:
        probs = self.detector.infer_single(recent_returns).astype(np.float32)
        return np.concatenate([np.asarray(base_state, dtype=np.float32), probs])

    def act(self, observation: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        return self.agent.act(observation, **kwargs)

    def store(self, *args: Any, **kwargs: Any) -> None:
        self.agent.store(*args, **kwargs)

    def update(self) -> dict[str, float] | None:
        return self.agent.update()

    def gate_weights(self, observation: np.ndarray) -> np.ndarray | None:
        return self.agent.gate_weights(observation)

    def hidden_activations(self, observation: np.ndarray) -> np.ndarray:
        return self.agent.hidden_activations(observation)

    def save_checkpoint(self, directory: str | Path) -> None:
        self.agent.save_checkpoint(directory)
        torch.save(self.detector.state_dict(), Path(directory) / "detector.pt")

    def load_checkpoint(self, directory: str | Path, weights_only: bool = True) -> None:
        self.agent.load_checkpoint(directory, weights_only=weights_only)
        detector_path = Path(directory) / "detector.pt"
        if detector_path.exists():
            detector_state = torch.load(detector_path, map_location="cpu", weights_only=False)
            if isinstance(detector_state, dict):
                self.detector.load_state_dict(detector_state)

    @property
    def action_dim(self) -> int:
        return self.agent.action_dim

    @property
    def model(self) -> Any:
        return self.agent.model

    @property
    def buffer(self) -> Any:
        return self.agent.buffer

    def reset_context(self) -> None:
        if hasattr(self.agent, 'reset_context'):
            self.agent.reset_context()
