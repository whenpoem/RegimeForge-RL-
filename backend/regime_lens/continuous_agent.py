from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .actor_critic import (
    PPOActorCritic,
    RCMoEActorCritic,
    RCMoESACActorCritic,
    SACActorCritic,
    build_actor_critic,
)
from .config import AlgorithmType, AgentType, GateType, TrainingConfig, config_to_snapshot
from .context import build_temporal_context, initialise_context_history, append_context_state
from .dqn import SumTree


def _resolve_device(device: str | None, fallback: str) -> torch.device:
    chosen = device or fallback
    if chosen in (None, "auto"):
        return torch.device("cpu")
    return torch.device(chosen)


def _as_numpy_vector(value: np.ndarray | torch.Tensor, *, name: str) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    result = np.asarray(array, dtype=np.float32)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return result


def _as_float(value: float | np.ndarray | torch.Tensor, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().reshape(-1)
        if tensor.numel() != 1:
            raise ValueError(f"{name} must contain exactly one element.")
        return float(tensor.item())
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one element.")
    return float(array[0])


def _stack_or_empty(items: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not items:
        return np.empty((0, *shape), dtype=np.float32)
    return np.stack(items, axis=0).astype(np.float32, copy=False)


class ContinuousReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = float(reward)
        self.next_observations[self.index] = next_observation
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "continuous_replay",
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "index": self.index,
            "size": self.size,
            "observations": self.observations.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_observations": self.next_observations.copy(),
            "dones": self.dones.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("kind")) != "continuous_replay":
            raise ValueError("Checkpoint buffer kind does not match ContinuousReplayBuffer.")
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("Checkpoint replay capacity does not match the current agent.")
        if int(state.get("observation_dim", -1)) != self.observation_dim:
            raise ValueError("Checkpoint observation dimension does not match the current agent.")
        if int(state.get("action_dim", -1)) != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match the current agent.")

        observations = np.asarray(state["observations"], dtype=np.float32)
        actions = np.asarray(state["actions"], dtype=np.float32)
        rewards = np.asarray(state["rewards"], dtype=np.float32)
        next_observations = np.asarray(state["next_observations"], dtype=np.float32)
        dones = np.asarray(state["dones"], dtype=np.float32)

        if observations.shape != (self.capacity, self.observation_dim):
            raise ValueError("Checkpoint replay observations have an unexpected shape.")
        if actions.shape != (self.capacity, self.action_dim):
            raise ValueError("Checkpoint replay actions have an unexpected shape.")
        if next_observations.shape != (self.capacity, self.observation_dim):
            raise ValueError("Checkpoint replay next observations have an unexpected shape.")
        if rewards.shape != (self.capacity,) or dones.shape != (self.capacity,):
            raise ValueError("Checkpoint replay rewards or dones have unexpected shapes.")

        size = int(state.get("size", 0))
        index = int(state.get("index", 0))
        if not 0 <= size <= self.capacity:
            raise ValueError("Checkpoint replay size is invalid.")
        if not 0 <= index < self.capacity:
            raise ValueError("Checkpoint replay index is invalid.")

        self.observations[...] = observations
        self.actions[...] = actions
        self.rewards[...] = rewards
        self.next_observations[...] = next_observations
        self.dones[...] = dones
        self.size = size
        self.index = index


class ContinuousPrioritizedReplayBuffer:
    """Prioritized replay buffer for continuous observations and actions."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = float(reward)
        self.next_observations[self.index] = next_observation
        self.dones[self.index] = float(done)
        self.tree.add(self._max_priority ** self.alpha)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        beta: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        total = self.tree.total
        segment = total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = rng.uniform(low, high)
            leaf_idx = self.tree.sample(value)
            leaf_idx = max(0, min(leaf_idx, self.size - 1))
            indices[i] = leaf_idx
            priorities[i] = max(self.tree.get_priority(leaf_idx), self.epsilon)
        probs = priorities / (total + 1e-10)
        is_weights = (self.size * probs) ** (-beta)
        is_weights = is_weights / (is_weights.max() + 1e-10)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
            is_weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors, strict=True):
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, abs(float(td_error)) + self.epsilon)
            self.tree.update(int(idx), priority)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "continuous_prioritized",
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "max_priority": self._max_priority,
            "index": self.index,
            "size": self.size,
            "observations": self.observations.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_observations": self.next_observations.copy(),
            "dones": self.dones.copy(),
            "tree": self.tree.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("kind")) != "continuous_prioritized":
            raise ValueError("Checkpoint buffer kind does not match ContinuousPrioritizedReplayBuffer.")
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("Checkpoint replay capacity does not match the current agent.")
        if int(state.get("observation_dim", -1)) != self.observation_dim:
            raise ValueError("Checkpoint observation dimension does not match the current agent.")
        if int(state.get("action_dim", -1)) != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match the current agent.")
        observations = np.asarray(state["observations"], dtype=np.float32)
        actions = np.asarray(state["actions"], dtype=np.float32)
        rewards = np.asarray(state["rewards"], dtype=np.float32)
        next_observations = np.asarray(state["next_observations"], dtype=np.float32)
        dones = np.asarray(state["dones"], dtype=np.float32)
        if observations.shape != (self.capacity, self.observation_dim):
            raise ValueError("Checkpoint replay observations have an unexpected shape.")
        if actions.shape != (self.capacity, self.action_dim):
            raise ValueError("Checkpoint replay actions have an unexpected shape.")
        if next_observations.shape != (self.capacity, self.observation_dim):
            raise ValueError("Checkpoint replay next observations have an unexpected shape.")
        if rewards.shape != (self.capacity,) or dones.shape != (self.capacity,):
            raise ValueError("Checkpoint replay rewards or dones have unexpected shapes.")
        size = int(state.get("size", 0))
        index = int(state.get("index", 0))
        if not 0 <= size <= self.capacity:
            raise ValueError("Checkpoint replay size is invalid.")
        if not 0 <= index < self.capacity:
            raise ValueError("Checkpoint replay index is invalid.")
        self.observations[...] = observations
        self.actions[...] = actions
        self.rewards[...] = rewards
        self.next_observations[...] = next_observations
        self.dones[...] = dones
        self.size = size
        self.index = index
        self.alpha = float(state.get("alpha", self.alpha))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self._max_priority = float(state.get("max_priority", self._max_priority))
        tree_state = state.get("tree")
        if isinstance(tree_state, dict):
            self.tree.load_state_dict(tree_state)


class RolloutBuffer:
    def __init__(self, observation_dim: int, action_dim: int):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.clear()

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.next_observations: list[np.ndarray] = []
        self.dones: list[float] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.next_observations.append(next_observation)
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "rollout",
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "observations": _stack_or_empty(self.observations, (self.observation_dim,)),
            "actions": _stack_or_empty(self.actions, (self.action_dim,)),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "next_observations": _stack_or_empty(self.next_observations, (self.observation_dim,)),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("kind")) != "rollout":
            raise ValueError("Checkpoint buffer kind does not match RolloutBuffer.")
        if int(state.get("observation_dim", -1)) != self.observation_dim:
            raise ValueError("Checkpoint observation dimension does not match the current agent.")
        if int(state.get("action_dim", -1)) != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match the current agent.")

        observations = np.asarray(state["observations"], dtype=np.float32)
        actions = np.asarray(state["actions"], dtype=np.float32)
        rewards = np.asarray(state["rewards"], dtype=np.float32)
        next_observations = np.asarray(state["next_observations"], dtype=np.float32)
        dones = np.asarray(state["dones"], dtype=np.float32)
        log_probs = np.asarray(state["log_probs"], dtype=np.float32)
        values = np.asarray(state["values"], dtype=np.float32)

        size = len(rewards)
        if observations.shape != (size, self.observation_dim):
            raise ValueError("Checkpoint rollout observations have an unexpected shape.")
        if actions.shape != (size, self.action_dim):
            raise ValueError("Checkpoint rollout actions have an unexpected shape.")
        if next_observations.shape != (size, self.observation_dim):
            raise ValueError("Checkpoint rollout next observations have an unexpected shape.")
        if dones.shape != (size,) or log_probs.shape != (size,) or values.shape != (size,):
            raise ValueError("Checkpoint rollout metadata has unexpected shapes.")

        self.observations = [row.copy() for row in observations]
        self.actions = [row.copy() for row in actions]
        self.rewards = rewards.astype(np.float32, copy=False).tolist()
        self.next_observations = [row.copy() for row in next_observations]
        self.dones = dones.astype(np.float32, copy=False).tolist()
        self.log_probs = log_probs.astype(np.float32, copy=False).tolist()
        self.values = values.astype(np.float32, copy=False).tolist()


class ContinuousActorCriticAgent:
    def __init__(
        self,
        config: TrainingConfig,
        *,
        observation_dim: int,
        action_dim: int,
        device: str | None = None,
        variant: str | None = None,
        ppo_epochs: int = 4,
        ppo_clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        sac_alpha: float = 0.2,
        target_entropy: float | None = None,
    ):
        if config.algorithm not in {AlgorithmType.PPO, AlgorithmType.SAC}:
            raise ValueError("ContinuousActorCriticAgent only supports PPO and SAC.")

        self.config = config
        self.algorithm = config.algorithm
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = _resolve_device(device, config.device)
        self.variant = self._resolve_variant(variant)
        self.ppo_epochs = max(int(ppo_epochs), 1)
        self.ppo_clip_eps = float(ppo_clip_eps)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.target_entropy = (
            float(target_entropy)
            if target_entropy is not None
            else -float(action_dim)
        )
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        self._use_temporal_context = (
            config.agent_type == AgentType.RCMOE_DQN
            and config.gate_type == GateType.TEMPORAL
            and not config.hierarchical_moe
        )
        self._context_len = max(config.context_len, 1)
        self._context_history: Any = None  # Deque, initialised on first act()

        self.model = build_actor_critic(
            config,
            observation_dim=observation_dim,
            action_dim=action_dim,
            device=str(self.device),
            variant=self.variant,
        )
        self.buffer: RolloutBuffer | ContinuousReplayBuffer | ContinuousPrioritizedReplayBuffer
        self.optimizer: torch.optim.Optimizer | None = None
        self.actor_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.alpha_optimizer: torch.optim.Optimizer | None = None
        self.target_model: SACActorCritic | RCMoESACActorCritic | None = None
        self.log_alpha: torch.Tensor | None = None

        if self.algorithm == AlgorithmType.PPO:
            if not isinstance(self.model, (PPOActorCritic, RCMoEActorCritic)):
                raise ValueError("PPO continuous agent requires a PPO-style actor-critic.")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
            self.buffer = RolloutBuffer(observation_dim, action_dim)
        else:
            if not isinstance(self.model, (SACActorCritic, RCMoESACActorCritic)):
                raise ValueError("SAC continuous agent requires the SAC actor-critic model.")
            actor_parameters = self.model.actor_parameters()
            critic_parameters = self.model.critic_parameters()
            self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=config.learning_rate)
            self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.learning_rate)
            self.log_alpha = torch.tensor(np.log(float(sac_alpha)), dtype=torch.float32, device=self.device)
            self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
            self.target_model = deepcopy(self.model).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            for parameter in self.target_model.parameters():
                parameter.requires_grad_(False)
            if config.use_per:
                self.buffer = ContinuousPrioritizedReplayBuffer(
                    config.replay_capacity, observation_dim, action_dim,
                    alpha=config.per_alpha, epsilon=config.per_epsilon,
                )
            else:
                self.buffer = ContinuousReplayBuffer(config.replay_capacity, observation_dim, action_dim)

    def _build_model_input(self, observation: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Build the input for the model, handling temporal context if needed."""
        if not self._use_temporal_context:
            return observation
        raw = observation if isinstance(observation, np.ndarray) else observation.detach().cpu().numpy()
        if self._context_history is None:
            self._context_history = initialise_context_history(raw, context_len=self._context_len)
        else:
            self._context_history = append_context_state(self._context_history, raw)
        return build_temporal_context(self._context_history, context_len=self._context_len)

    def act(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        model_input = self._build_model_input(observation)
        with torch.inference_mode():
            outputs = self.model.act(model_input, deterministic=deterministic)
        if isinstance(observation, torch.Tensor):
            return outputs
        converted: dict[str, np.ndarray] = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                converted[key] = value.detach().cpu().numpy()
            else:
                converted[key] = np.asarray(value)
        return converted

    def store(
        self,
        observation: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor,
        reward: float,
        next_observation: np.ndarray | torch.Tensor,
        done: bool,
        *,
        log_prob: float | np.ndarray | torch.Tensor | None = None,
        value: float | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        observation_array = _as_numpy_vector(observation, name="observation")
        action_array = _as_numpy_vector(action, name="action")
        next_observation_array = _as_numpy_vector(next_observation, name="next_observation")

        if observation_array.shape[0] != self.observation_dim:
            raise ValueError("Observation shape does not match the configured observation_dim.")
        if next_observation_array.shape[0] != self.observation_dim:
            raise ValueError("Next observation shape does not match the configured observation_dim.")
        if action_array.shape[0] != self.action_dim:
            raise ValueError("Action shape does not match the configured action_dim.")

        if self.algorithm == AlgorithmType.PPO:
            if log_prob is None or value is None:
                with torch.inference_mode():
                    evaluation = self.model.evaluate_actions(
                        torch.as_tensor(observation_array, dtype=torch.float32, device=self.device).unsqueeze(0),
                        torch.as_tensor(action_array, dtype=torch.float32, device=self.device).unsqueeze(0),
                    )
                if log_prob is None:
                    log_prob = evaluation["log_prob"]
                if value is None:
                    value = evaluation["value"]
            assert isinstance(self.buffer, RolloutBuffer)
            self.buffer.add(
                observation_array,
                action_array,
                float(reward),
                next_observation_array,
                bool(done),
                _as_float(log_prob, name="log_prob"),
                _as_float(value, name="value"),
            )
            return

        assert isinstance(self.buffer, (ContinuousReplayBuffer, ContinuousPrioritizedReplayBuffer))
        self.buffer.add(
            observation_array,
            action_array,
            float(reward),
            next_observation_array,
            bool(done),
        )

    def update(self) -> dict[str, float] | None:
        if self.algorithm == AlgorithmType.PPO:
            return self._update_ppo()
        return self._update_sac()

    def hidden_activations(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        obs = _as_numpy_vector(observation, name="observation")
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            if hasattr(self.model, "encoder"):
                hidden = self.model.encoder(tensor)
            else:
                outputs = self.model.forward(tensor)
                hidden = outputs.get("mean", tensor)
        return hidden.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def reset_context(self) -> None:
        """Reset the temporal context history (call at episode boundaries)."""
        self._context_history = None

    def gate_weights(self, observation: np.ndarray | torch.Tensor) -> np.ndarray | None:
        obs = _as_numpy_vector(observation, name="observation")
        model_input = self._build_model_input(obs)
        tensor = torch.as_tensor(model_input, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model.forward(tensor)
        gate = outputs.get("gate_weights") if isinstance(outputs, dict) else None
        if gate is None:
            return None
        return gate.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_dir / "online.pt")
        if self.target_model is not None:
            torch.save(self.target_model.state_dict(), checkpoint_dir / "target.pt")
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        if self.actor_optimizer is not None:
            torch.save(self.actor_optimizer.state_dict(), checkpoint_dir / "actor_optimizer.pt")
        if self.critic_optimizer is not None:
            torch.save(self.critic_optimizer.state_dict(), checkpoint_dir / "critic_optimizer.pt")
        if self.alpha_optimizer is not None:
            torch.save(self.alpha_optimizer.state_dict(), checkpoint_dir / "alpha_optimizer.pt")
        torch.save(self._checkpoint_metadata(), checkpoint_dir / "meta.pt")

    def load_checkpoint(self, path: str | Path, weights_only: bool = False) -> None:
        checkpoint_dir = Path(path)
        online_path = checkpoint_dir / "online.pt"
        if not online_path.exists():
            raise FileNotFoundError(f"No model checkpoint found at {online_path}")
        self.model.load_state_dict(
            torch.load(online_path, map_location=self.device, weights_only=True)
        )

        if self.target_model is not None:
            target_path = checkpoint_dir / "target.pt"
            if target_path.exists():
                self.target_model.load_state_dict(
                    torch.load(target_path, map_location=self.device, weights_only=True)
                )
            else:
                self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

        if weights_only:
            return

        if self.optimizer is not None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            if optimizer_path.exists():
                self.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=self.device, weights_only=True)
                )
        if self.actor_optimizer is not None:
            actor_optimizer_path = checkpoint_dir / "actor_optimizer.pt"
            if actor_optimizer_path.exists():
                self.actor_optimizer.load_state_dict(
                    torch.load(actor_optimizer_path, map_location=self.device, weights_only=True)
                )
        if self.critic_optimizer is not None:
            critic_optimizer_path = checkpoint_dir / "critic_optimizer.pt"
            if critic_optimizer_path.exists():
                self.critic_optimizer.load_state_dict(
                    torch.load(critic_optimizer_path, map_location=self.device, weights_only=True)
                )
        if self.alpha_optimizer is not None:
            alpha_optimizer_path = checkpoint_dir / "alpha_optimizer.pt"
            if alpha_optimizer_path.exists():
                self.alpha_optimizer.load_state_dict(
                    torch.load(alpha_optimizer_path, map_location=self.device, weights_only=True)
                )

        meta_path = checkpoint_dir / "meta.pt"
        if meta_path.exists():
            metadata = torch.load(meta_path, map_location="cpu", weights_only=False)
            if isinstance(metadata, dict):
                self._restore_checkpoint_metadata(metadata)

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is None:
            return torch.tensor(0.0, device=self.device)
        return self.log_alpha.exp()

    def _resolve_variant(self, variant: str | None) -> str | None:
        if variant is not None:
            return variant
        if self.config.agent_type == AgentType.RCMOE_DQN:
            return "rcmoe"
        if self.algorithm == AlgorithmType.SAC:
            return "sac"
        return "ppo"

    def _checkpoint_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "config": config_to_snapshot(self.config),
            "algorithm": self.algorithm.value,
            "variant": self.variant,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "use_temporal_context": self._use_temporal_context,
            "context_len": self._context_len,
            "ppo_epochs": self.ppo_epochs,
            "ppo_clip_eps": self.ppo_clip_eps,
            "gae_lambda": self.gae_lambda,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "target_entropy": self.target_entropy,
            "rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() and self.device.type == "cuda"
                else None
            ),
            "buffer": self.buffer.state_dict(),
        }
        if self.log_alpha is not None:
            metadata["log_alpha"] = self.log_alpha.detach().cpu()
        return metadata

    def _restore_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        if str(metadata.get("algorithm", self.algorithm.value)) != self.algorithm.value:
            raise ValueError("Checkpoint algorithm does not match the current agent.")
        if int(metadata.get("observation_dim", self.observation_dim)) != self.observation_dim:
            raise ValueError("Checkpoint observation dimension does not match the current agent.")
        if int(metadata.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match the current agent.")
        if bool(metadata.get("use_temporal_context", self._use_temporal_context)) != self._use_temporal_context:
            raise ValueError("Checkpoint temporal context setting does not match the current agent.")

        buffer_state = metadata.get("buffer")
        if isinstance(buffer_state, dict):
            self.buffer.load_state_dict(buffer_state)

        rng_state = metadata.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        torch_rng_state = metadata.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state)
        cuda_rng_state = metadata.get("cuda_rng_state")
        if (
            cuda_rng_state is not None
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        ):
            torch.cuda.set_rng_state_all(cuda_rng_state)

        if self.log_alpha is not None and "log_alpha" in metadata:
            log_alpha_value = torch.as_tensor(metadata["log_alpha"], dtype=torch.float32, device=self.device)
            self.log_alpha.data.copy_(log_alpha_value.reshape_as(self.log_alpha))

    def _update_ppo(self) -> dict[str, float] | None:
        assert isinstance(self.buffer, RolloutBuffer)
        assert self.optimizer is not None
        if len(self.buffer) == 0:
            return None

        observations = torch.as_tensor(
            np.stack(self.buffer.observations, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.stack(self.buffer.actions, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(self.buffer.values, dtype=torch.float32, device=self.device)

        last_done = bool(self.buffer.dones[-1])
        if last_done:
            bootstrap_value = torch.zeros((), dtype=torch.float32, device=self.device)
        else:
            last_next_observation = torch.as_tensor(
                self.buffer.next_observations[-1],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            with torch.inference_mode():
                bootstrap_value = self.model.forward(last_next_observation)["value"].reshape(())

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        for index in range(len(self.buffer) - 1, -1, -1):
            if index == len(self.buffer) - 1:
                next_value = bootstrap_value
            else:
                next_value = values[index + 1]
            mask = 1.0 - dones[index]
            delta = rewards[index] + self.config.gamma * mask * next_value - values[index]
            gae = delta + self.config.gamma * self.gae_lambda * mask * gae
            advantages[index] = gae

        returns = advantages + values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        minibatch_size = min(self.config.batch_size, len(self.buffer))
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "load_balance_loss": 0.0,
            "total_loss": 0.0,
        }
        update_count = 0

        for _ in range(self.ppo_epochs):
            indices = self.rng.permutation(len(self.buffer))
            for start in range(0, len(self.buffer), minibatch_size):
                batch_indices = indices[start : start + minibatch_size]
                batch_observations = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                evaluation = self.model.evaluate_actions(batch_observations, batch_actions)
                new_log_probs = evaluation["log_prob"].squeeze(-1)
                entropy = evaluation["entropy"].squeeze(-1)
                new_values = evaluation["value"].squeeze(-1)

                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.ppo_clip_eps,
                    1.0 + self.ppo_clip_eps,
                ) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(new_values, batch_returns)
                entropy_bonus = entropy.mean()

                load_balance_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if "gate_weights" in evaluation and self.config.load_balance_weight > 0.0:
                    gate_weights = evaluation["gate_weights"]
                    target_mass = 1.0 / gate_weights.shape[-1]
                    load_balance_loss = (gate_weights.mean(dim=0) - target_mass).pow(2).mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                    + self.config.load_balance_weight * load_balance_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_bonus.item())
                metrics["load_balance_loss"] += float(load_balance_loss.item())
                metrics["total_loss"] += float(loss.item())
                update_count += 1

        self.buffer.clear()
        return {
            key: value / max(update_count, 1)
            for key, value in metrics.items()
        } | {"updates": float(update_count)}

    def _update_sac(self) -> dict[str, float] | None:
        assert isinstance(self.buffer, (ContinuousReplayBuffer, ContinuousPrioritizedReplayBuffer))
        assert isinstance(self.model, (SACActorCritic, RCMoESACActorCritic))
        assert isinstance(self.target_model, (SACActorCritic, RCMoESACActorCritic))
        assert self.actor_optimizer is not None
        assert self.critic_optimizer is not None
        assert self.alpha_optimizer is not None
        assert self.log_alpha is not None

        if len(self.buffer) < self.config.batch_size:
            return None

        use_per = isinstance(self.buffer, ContinuousPrioritizedReplayBuffer)
        actor_parameters = self.model.actor_parameters()
        critic_parameters = self.model.critic_parameters()
        critic_freeze_parameters = self.model.critic_freeze_parameters()
        metrics = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "alpha": 0.0,
            "load_balance_loss": 0.0,
        }
        updates = 0

        for _ in range(max(int(self.config.gradient_steps), 1)):
            if use_per:
                observations_np, actions_np, rewards_np, next_observations_np, dones_np, is_weights_np, tree_indices = self.buffer.sample(
                    self.config.batch_size,
                    self.rng,
                    beta=self.config.per_beta_start,
                )
                is_weights = torch.as_tensor(is_weights_np, dtype=torch.float32, device=self.device)
            else:
                observations_np, actions_np, rewards_np, next_observations_np, dones_np = self.buffer.sample(
                    self.config.batch_size,
                    self.rng,
                )
            observations = torch.as_tensor(observations_np, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
            rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device).unsqueeze(-1)
            next_observations = torch.as_tensor(
                next_observations_np,
                dtype=torch.float32,
                device=self.device,
            )
            dones = torch.as_tensor(dones_np, dtype=torch.float32, device=self.device).unsqueeze(-1)

            with torch.no_grad():
                next_sample = self.model.sample_action(next_observations)
                target_q1 = self.target_model.q1(next_observations, next_sample.action)
                target_q2 = self.target_model.q2(next_observations, next_sample.action)
                target_value = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_sample.log_prob
                target_q = rewards + (1.0 - dones) * self.config.gamma * target_value

            current_q1 = self.model.q1(observations, actions)
            current_q2 = self.model.q2(observations, actions)
            td_error_q1 = (current_q1 - target_q).abs()
            td_error_q2 = (current_q2 - target_q).abs()
            if use_per:
                critic_loss = (
                    (td_error_q1.pow(2) * is_weights.unsqueeze(-1)).mean()
                    + (td_error_q2.pow(2) * is_weights.unsqueeze(-1)).mean()
                )
            else:
                critic_loss = (
                    torch.nn.functional.mse_loss(current_q1, target_q)
                    + torch.nn.functional.mse_loss(current_q2, target_q)
                )

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic_parameters, max_norm=1.0)
            self.critic_optimizer.step()

            for parameter in critic_freeze_parameters:
                parameter.requires_grad_(False)

            policy_sample = self.model.sample_action(observations)
            q1_policy = self.model.q1(observations, policy_sample.action)
            q2_policy = self.model.q2(observations, policy_sample.action)
            load_balance_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            if self.config.load_balance_weight > 0.0:
                gate_weights = policy_sample.extras.get("gate_weights")
                if gate_weights is not None:
                    target_mass = 1.0 / gate_weights.shape[-1]
                    load_balance_loss = (gate_weights.mean(dim=0) - target_mass).pow(2).mean()
            actor_loss = (
                self.alpha.detach() * policy_sample.log_prob - torch.minimum(q1_policy, q2_policy)
            ).mean() + self.config.load_balance_weight * load_balance_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor_parameters, max_norm=1.0)
            self.actor_optimizer.step()

            for parameter in critic_freeze_parameters:
                parameter.requires_grad_(True)

            alpha_loss = -(
                self.log_alpha * (policy_sample.log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self._soft_update_target()

            if use_per:
                with torch.no_grad():
                    raw_td = ((td_error_q1 + td_error_q2) / 2.0).squeeze(-1).cpu().numpy()
                self.buffer.update_priorities(tree_indices, raw_td)

            metrics["critic_loss"] += float(critic_loss.item())
            metrics["actor_loss"] += float(actor_loss.item())
            metrics["alpha_loss"] += float(alpha_loss.item())
            metrics["alpha"] += float(self.alpha.detach().item())
            metrics["load_balance_loss"] += float(load_balance_loss.item())
            updates += 1

        return {
            key: value / max(updates, 1)
            for key, value in metrics.items()
        } | {"updates": float(updates)}

    def _soft_update_target(self) -> None:
        assert self.target_model is not None
        for target_parameter, parameter in zip(
            self.target_model.parameters(),
            self.model.parameters(),
            strict=True,
        ):
            target_parameter.data.mul_(1.0 - self.config.tau).add_(parameter.data, alpha=self.config.tau)


__all__ = [
    "ContinuousActorCriticAgent",
    "ContinuousPrioritizedReplayBuffer",
    "ContinuousReplayBuffer",
    "RolloutBuffer",
]
