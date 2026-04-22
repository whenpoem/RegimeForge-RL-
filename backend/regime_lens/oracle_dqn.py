"""Oracle DQN — upper-bound baseline that observes true regime.

The Oracle agent receives the ground-truth regime as a one-hot vector
appended to the standard observation.  This represents the theoretical
best a DQN can do when regime information is *directly observable*,
serving as a performance ceiling for the RCMoE-DQN whose gate network
must *infer* the regime from price features alone.
"""

from __future__ import annotations

import numpy as np

from .config import REGIME_LABELS
from .dqn import DQNAgent


class OracleDQNAgent(DQNAgent):
    """DQN agent that receives regime one-hot as part of its observation.

    Wraps the standard ``DQNAgent`` — the only difference is the
    observation dimension is ``base_obs_dim + n_regimes``.
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
        **kwargs,
    ):
        n_regimes = len(REGIME_LABELS)
        super().__init__(
            observation_dim=base_observation_dim + n_regimes,
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
        self._n_regimes = n_regimes
        self._regime_index = {label: i for i, label in enumerate(REGIME_LABELS)}

    # -- Helpers to build oracle observations --------------------------------

    def augment_state(self, base_state: np.ndarray, regime_label: str) -> np.ndarray:
        """Append regime one-hot to the base observation vector."""
        onehot = np.zeros(self._n_regimes, dtype=np.float32)
        idx = self._regime_index.get(regime_label, -1)
        if 0 <= idx < self._n_regimes:
            onehot[idx] = 1.0
        return np.concatenate([np.asarray(base_state, dtype=np.float32), onehot])
