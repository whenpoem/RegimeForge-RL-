from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import REGIME_LABELS, AgentType, GateType, TrainingConfig
from .context import append_context_state, build_temporal_context, initialise_context_history
from .continuous_agent import ContinuousActorCriticAgent
from .dqn import DQNAgent
from .hmm_dqn import HMMContinuousAgent, HMMDQNAgent
from .oracle_dqn import OracleDQNAgent
from .rcmoe import RCMoEAgent
from .transformer_agent import TransformerDQNAgent
from .world_model import WorldModelAgent


Agent = DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent | ContinuousActorCriticAgent | HMMContinuousAgent | TransformerDQNAgent | WorldModelAgent


def is_rcmoe(agent: object) -> bool:
    return isinstance(agent, RCMoEAgent)


def is_oracle(agent: object) -> bool:
    return isinstance(agent, OracleDQNAgent)


def is_hmm(agent: object) -> bool:
    return isinstance(agent, HMMDQNAgent)


def is_hmm_continuous(agent: object) -> bool:
    return isinstance(agent, HMMContinuousAgent)


def is_transformer(agent: object) -> bool:
    return isinstance(agent, TransformerDQNAgent)


def is_world_model(agent: object) -> bool:
    return isinstance(agent, WorldModelAgent)


def is_continuous_agent(agent: object) -> bool:
    return isinstance(agent, ContinuousActorCriticAgent)


def reset_agent_context(agent: object) -> None:
    reset = getattr(agent, "reset_context", None)
    if callable(reset):
        reset()


def _uniform_regime() -> np.ndarray:
    return np.full(len(REGIME_LABELS), 1.0 / len(REGIME_LABELS), dtype=np.float32)


def _uniform_components(agent: object) -> np.ndarray:
    n_components = int(getattr(agent, "_n_components", len(REGIME_LABELS)))
    return np.full(n_components, 1.0 / n_components, dtype=np.float32)


def _augment_oracle(state: np.ndarray) -> np.ndarray:
    return np.concatenate([state, _uniform_regime()]).astype(np.float32)


def _augment_hmm_neutral(agent: object, state: np.ndarray) -> np.ndarray:
    return np.concatenate([state, _uniform_components(agent)]).astype(np.float32)


@dataclass(slots=True)
class AgentObservationAdapter:
    """Centralizes environment-to-agent observation contracts.

    Training code should not know how Oracle, HMM, Transformer, or temporal-gate
    agents encode state. Keeping that mapping here prevents train/eval/surface
    paths from drifting apart as new agents are added.
    """

    config: TrainingConfig
    agent: Agent
    context_history: Any = None

    def reset_episode(self, env: Any, initial_state: np.ndarray) -> None:
        reset_agent_context(self.agent)
        if is_hmm(self.agent) or is_hmm_continuous(self.agent):
            self.agent.fit_detector(env.returns[: self.config.warmup_steps])
        self.context_history = initialise_context_history(
            np.asarray(initial_state, dtype=np.float32),
            context_len=self.context_len,
        )

    @property
    def context_len(self) -> int:
        return max(int(getattr(self.agent, "context_len", self.config.context_len)), 1)

    @property
    def uses_discrete_temporal_gate(self) -> bool:
        return is_rcmoe(self.agent) and getattr(self.agent, "gate_type", self.config.gate_type) == GateType.TEMPORAL

    def discrete_observation(self, env: Any, state: np.ndarray) -> np.ndarray:
        if is_oracle(self.agent):
            return env.observe_oracle().state
        if is_hmm(self.agent):
            return self.agent.augment_state(
                np.asarray(state, dtype=np.float32),
                env.returns[max(0, env.t - 10) : env.t],
            )
        if self.uses_discrete_temporal_gate:
            return build_temporal_context(self.context_history, context_len=self.context_len)
        return np.asarray(state, dtype=np.float32)

    def next_discrete_observation(self, env: Any, next_state: np.ndarray) -> np.ndarray:
        if is_oracle(self.agent):
            return env.observe_oracle().state
        if is_hmm(self.agent):
            return self.agent.augment_state(
                np.asarray(next_state, dtype=np.float32),
                env.returns[max(0, env.t - 10) : env.t],
            )
        if self.uses_discrete_temporal_gate:
            next_history = append_context_state(self.context_history, next_state)
            return build_temporal_context(next_history, context_len=self.context_len)
        return np.asarray(next_state, dtype=np.float32)

    def advance(self, next_state: np.ndarray) -> None:
        if self.context_history is not None:
            self.context_history = append_context_state(self.context_history, next_state)

    def continuous_observation(self, env: Any, state: np.ndarray) -> np.ndarray:
        if self.config.agent_type == AgentType.ORACLE_DQN:
            return env.observe_oracle().state
        if is_hmm_continuous(self.agent):
            return self.agent.augment_state(
                np.asarray(state, dtype=np.float32),
                env.returns[max(0, env.t - 10) : env.t],
            )
        return np.asarray(state, dtype=np.float32)

    def neutral_surface_observation(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if self.config.agent_type == AgentType.ORACLE_DQN:
            return _augment_oracle(state)
        if is_hmm(self.agent) or is_hmm_continuous(self.agent):
            return _augment_hmm_neutral(self.agent, state)
        return state

    def explainability_observation(self, base_state: np.ndarray) -> np.ndarray:
        state = self.neutral_surface_observation(base_state)
        if is_transformer(self.agent):
            seq_len = max(int(getattr(self.agent, "seq_len", self.config.seq_len)), 1)
            return np.tile(state, seq_len).astype(np.float32)
        if self.uses_discrete_temporal_gate:
            return np.tile(state, self.context_len).astype(np.float32)
        return state

    def batch_policy_observations(self, states: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        base_states = np.asarray(states, dtype=np.float32)
        observations = np.asarray(
            [self.neutral_surface_observation(state) for state in base_states],
            dtype=np.float32,
        )
        if is_transformer(self.agent):
            seq_len = max(int(getattr(self.agent, "seq_len", self.config.seq_len)), 1)
            return np.asarray([np.tile(state, seq_len) for state in observations], dtype=np.float32)
        if self.uses_discrete_temporal_gate:
            return np.asarray([np.tile(state, self.context_len) for state in observations], dtype=np.float32)
        return observations


__all__ = [
    "Agent",
    "AgentObservationAdapter",
    "is_continuous_agent",
    "is_hmm",
    "is_hmm_continuous",
    "is_oracle",
    "is_rcmoe",
    "is_transformer",
    "is_world_model",
    "reset_agent_context",
]
