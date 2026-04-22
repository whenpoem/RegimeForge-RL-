from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int):
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.states = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_states = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "uniform",
            **self._storage_state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("kind", "uniform")) != "uniform":
            raise ValueError("Checkpoint buffer kind does not match ReplayBuffer.")
        self._restore_storage_state(state)

    def _storage_state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "index": self.index,
            "size": self.size,
            "states": self.states.copy(),
            "next_states": self.next_states.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "dones": self.dones.copy(),
        }

    def _restore_storage_state(self, state: dict[str, Any]) -> None:
        capacity = int(state.get("capacity", -1))
        observation_dim = int(state.get("observation_dim", -1))
        if capacity != self.capacity or observation_dim != self.observation_dim:
            raise ValueError(
                "Checkpoint replay buffer shape does not match the current agent configuration."
            )

        states = np.asarray(state["states"], dtype=np.float32)
        next_states = np.asarray(state["next_states"], dtype=np.float32)
        actions = np.asarray(state["actions"], dtype=np.int64)
        rewards = np.asarray(state["rewards"], dtype=np.float32)
        dones = np.asarray(state["dones"], dtype=np.float32)

        expected_state_shape = (self.capacity, self.observation_dim)
        if states.shape != expected_state_shape or next_states.shape != expected_state_shape:
            raise ValueError("Checkpoint replay buffer tensors have unexpected shapes.")
        if actions.shape != (self.capacity,) or rewards.shape != (self.capacity,) or dones.shape != (self.capacity,):
            raise ValueError("Checkpoint replay buffer arrays have unexpected shapes.")

        size = int(state.get("size", 0))
        index = int(state.get("index", 0))
        if not 0 <= size <= self.capacity:
            raise ValueError("Checkpoint replay buffer size is invalid.")
        if not 0 <= index < self.capacity:
            raise ValueError("Checkpoint replay buffer index is invalid.")

        self.states[...] = states
        self.next_states[...] = next_states
        self.actions[...] = actions
        self.rewards[...] = rewards
        self.dones[...] = dones
        self.size = size
        self.index = index


# ---------------------------------------------------------------------------
# Sum-tree for O(log n) proportional sampling
# ---------------------------------------------------------------------------

class SumTree:
    """Binary tree where each leaf stores a priority value.

    Internal nodes store the sum of their children, enabling O(log n)
    proportional sampling and O(log n) priority updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_index = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def update(self, leaf_index: int, priority: float) -> None:
        """Set priority for a leaf and propagate change upward."""
        tree_index = leaf_index + self.capacity - 1
        delta = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index > 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += delta

    def add(self, priority: float) -> int:
        """Add a new priority (overwrites the oldest if full). Returns leaf index."""
        leaf_index = self.write_index
        self.update(leaf_index, priority)
        self.write_index = (self.write_index + 1) % self.capacity
        return leaf_index

    def sample(self, value: float) -> int:
        """Sample a leaf index proportional to stored priorities."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)

    def get_priority(self, leaf_index: int) -> float:
        return float(self.tree[leaf_index + self.capacity - 1])

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "tree": self.tree.copy(),
            "write_index": self.write_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        capacity = int(state.get("capacity", -1))
        if capacity != self.capacity:
            raise ValueError("Checkpoint sum-tree capacity does not match the current buffer.")
        tree = np.asarray(state["tree"], dtype=np.float64)
        expected_shape = (2 * self.capacity - 1,)
        if tree.shape != expected_shape:
            raise ValueError("Checkpoint sum-tree has an unexpected shape.")
        write_index = int(state.get("write_index", 0))
        if not 0 <= write_index < self.capacity:
            raise ValueError("Checkpoint sum-tree write index is invalid.")
        self.tree[...] = tree
        self.write_index = write_index


class PrioritizedReplayBuffer:
    """Experience replay buffer with proportional prioritization.

    Implements the proportional variant from
    *Prioritized Experience Replay* (Schaul et al., 2016).

    Key features:
    - O(log n) sampling via SumTree
    - Importance-sampling weights for bias correction
    - Priority proportional to |TD-error|^alpha
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

        self.states = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_states = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

        self.index = 0
        self.size = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = float(done)
        # New transitions get max priority so they are sampled at least once
        self.tree.add(self._max_priority ** self.alpha)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        beta: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a prioritized batch.

        Returns:
            (states, actions, rewards, next_states, dones, is_weights, indices)
            is_weights: importance-sampling weights for bias correction.
            indices: leaf indices for later priority updates.
        """
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        total = self.tree.total

        segment = total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = rng.uniform(low, high)
            leaf_idx = self.tree.sample(value)
            # Clamp to valid range
            leaf_idx = max(0, min(leaf_idx, self.size - 1))
            indices[i] = leaf_idx
            priorities[i] = max(self.tree.get_priority(leaf_idx), self.epsilon)

        # Importance-sampling weights
        probs = priorities / (total + 1e-10)
        is_weights = (self.size * probs) ** (-beta)
        is_weights = is_weights / (is_weights.max() + 1e-10)  # normalize

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            is_weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors, strict=True):
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, abs(float(td_error)) + self.epsilon)
            self.tree.update(int(idx), priority)

    def _storage_state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "index": self.index,
            "size": self.size,
            "states": self.states.copy(),
            "next_states": self.next_states.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "dones": self.dones.copy(),
        }

    def _restore_storage_state(self, state: dict[str, Any]) -> None:
        capacity = int(state.get("capacity", -1))
        observation_dim = int(state.get("observation_dim", -1))
        if capacity != self.capacity or observation_dim != self.observation_dim:
            raise ValueError(
                "Checkpoint replay buffer shape does not match the current agent configuration."
            )

        states = np.asarray(state["states"], dtype=np.float32)
        next_states = np.asarray(state["next_states"], dtype=np.float32)
        actions = np.asarray(state["actions"], dtype=np.int64)
        rewards = np.asarray(state["rewards"], dtype=np.float32)
        dones = np.asarray(state["dones"], dtype=np.float32)

        expected_state_shape = (self.capacity, self.observation_dim)
        if states.shape != expected_state_shape or next_states.shape != expected_state_shape:
            raise ValueError("Checkpoint replay buffer tensors have unexpected shapes.")
        if actions.shape != (self.capacity,) or rewards.shape != (self.capacity,) or dones.shape != (self.capacity,):
            raise ValueError("Checkpoint replay buffer arrays have unexpected shapes.")

        size = int(state.get("size", 0))
        index = int(state.get("index", 0))
        if not 0 <= size <= self.capacity:
            raise ValueError("Checkpoint replay buffer size is invalid.")
        if not 0 <= index < self.capacity:
            raise ValueError("Checkpoint replay buffer index is invalid.")

        self.states[...] = states
        self.next_states[...] = next_states
        self.actions[...] = actions
        self.rewards[...] = rewards
        self.dones[...] = dones
        self.size = size
        self.index = index

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "prioritized",
            **self._storage_state_dict(),
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "max_priority": self._max_priority,
            "tree": self.tree.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("kind")) != "prioritized":
            raise ValueError("Checkpoint buffer kind does not match PrioritizedReplayBuffer.")
        self._restore_storage_state(state)
        self.alpha = float(state.get("alpha", self.alpha))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self._max_priority = float(state.get("max_priority", self._max_priority))
        tree_state = state.get("tree")
        if not isinstance(tree_state, dict):
            raise ValueError("Checkpoint prioritized replay metadata is missing the sum-tree state.")
        self.tree.load_state_dict(tree_state)


def save_agent_checkpoint_bundle(
    directory: str | Path,
    online: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    meta: dict[str, Any],
) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(online.state_dict(), d / "online.pt")
    torch.save(target.state_dict(), d / "target.pt")
    torch.save(optimizer.state_dict(), d / "optimizer.pt")
    torch.save(meta, d / "meta.pt")
    return d


def load_agent_checkpoint_bundle(
    directory: str | Path,
    device: torch.device,
    online: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    weights_only: bool,
) -> dict[str, Any]:
    d = Path(directory)
    online_path = d / "online.pt"
    if not online_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {online_path}")
    online.load_state_dict(
        torch.load(online_path, map_location=device, weights_only=True)
    )
    target_path = d / "target.pt"
    if target_path.exists():
        target.load_state_dict(
            torch.load(target_path, map_location=device, weights_only=True)
        )
    else:
        target.load_state_dict(online.state_dict())
    meta: dict[str, Any] = {}
    if not weights_only:
        opt_path = d / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=device, weights_only=True)
            )
        meta_path = d / "meta.pt"
        if meta_path.exists():
            meta_payload = torch.load(meta_path, map_location="cpu", weights_only=False)
            if isinstance(meta_payload, dict):
                meta = meta_payload
    return meta

class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        sigma = self.sigma_init / np.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self) -> None:
        device = self.weight_mu.device
        epsilon_in = self._scale_noise(self.in_features, device=device)
        epsilon_out = self._scale_noise(self.out_features, device=device)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(inputs, weight, bias)

    @staticmethod
    def _scale_noise(size: int, *, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        dueling: bool = False,
        noisy: bool = False,
    ):
        super().__init__()
        linear = NoisyLinear if noisy else nn.Linear
        self.dueling = dueling
        self.noisy = noisy
        self.trunk = nn.Sequential(
            linear(observation_dim, hidden_dim),
            nn.SiLU(),
            linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        if dueling:
            self.value_head = linear(hidden_dim, 1)
            self.advantage_head = linear(hidden_dim, action_dim)
        else:
            self.value_head = None
            self.advantage_head = None
            self.output_head = linear(hidden_dim, action_dim)

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.trunk(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(inputs)
        if self.dueling:
            assert self.value_head is not None
            assert self.advantage_head is not None
            value = self.value_head(hidden)
            advantage = self.advantage_head(hidden)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        return self.output_head(hidden)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DQNAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        learning_rate: float,
        gamma: float,
        tau: float,
        replay_capacity: int,
        batch_size: int,
        device: str,
        seed: int,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_epsilon: float = 1e-6,
        dueling: bool = False,
        noisy: bool = False,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_per = use_per
        self.dueling = dueling
        self.noisy = noisy
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.online = QNetwork(observation_dim, action_dim, hidden_dim, dueling=dueling, noisy=noisy).to(self.device)
        self.target = QNetwork(observation_dim, action_dim, hidden_dim, dueling=dueling, noisy=noisy).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # per-sample for PER
        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                replay_capacity, observation_dim, alpha=per_alpha, epsilon=per_epsilon,
            )
        else:
            self.buffer = ReplayBuffer(replay_capacity, observation_dim)
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if self.noisy:
            self.online.reset_noise()
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_dim))
        return self.greedy_action(state)

    def greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            values = self.online(state_tensor)
        return int(torch.argmax(values, dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            values = self.online(state_tensor).squeeze(0).cpu().numpy()
        return values.astype(np.float64)

    def batch_q_values(self, states: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            values = self.online(state_tensor).cpu().numpy()
        return values.astype(np.float64, copy=False)

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.add(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

    def update(self, per_beta: float = 0.4) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        if self.use_per:
            states_np, actions_np, rewards_np, next_states_np, dones_np, is_weights_np, tree_indices = \
                self.buffer.sample(self.batch_size, self.rng, beta=per_beta)
            is_weights = torch.from_numpy(is_weights_np).to(self.device, non_blocking=True)
        else:
            states_np, actions_np, rewards_np, next_states_np, dones_np = \
                self.buffer.sample(self.batch_size, self.rng)
            is_weights = None

        states = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions_np).to(self.device, non_blocking=True).unsqueeze(1)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)

        q_selected = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.noisy:
                self.online.reset_noise()
                self.target.reset_noise()
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        td_errors = self.loss_fn(q_selected, target_q)  # per-sample
        if is_weights is not None:
            loss = (td_errors * is_weights).mean()
        else:
            loss = td_errors.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._soft_update()

        # Update PER priorities with raw TD errors
        if self.use_per:
            with torch.no_grad():
                raw_td = (q_selected - target_q).abs().cpu().numpy()
            self.buffer.update_priorities(tree_indices, raw_td)

        return float(loss.item())

    def hidden_activations(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            hidden = self.online.hidden(state_tensor)
        return hidden.squeeze(0).cpu().numpy().astype(np.float64)

    def batch_hidden_activations(self, states: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            hidden = self.online.hidden(state_tensor)
        return hidden.cpu().numpy().astype(np.float64, copy=False)

    # -- Checkpoint persistence -----------------------------------------------

    def save_checkpoint(self, directory: str | Path) -> None:
        """Save model weights, optimizer state, RNG state, and replay state."""
        save_agent_checkpoint_bundle(
            directory,
            self.online,
            self.target,
            self.optimizer,
            self._checkpoint_metadata(),
        )

    def load_checkpoint(self, directory: str | Path, weights_only: bool = True) -> None:
        """Load model weights and optionally restore optimizer/replay state.

        Args:
            directory: Path to checkpoint directory containing .pt files.
            weights_only: If True, only load network weights (for eval).
                          If False, also restore optimizer, RNG, and replay state.
        """
        metadata = load_agent_checkpoint_bundle(
            directory,
            self.device,
            self.online,
            self.target,
            self.optimizer,
            weights_only=weights_only,
        )
        self.target.eval()
        if not weights_only:
            self._restore_checkpoint_metadata(metadata)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "use_per": self.use_per,
            "dueling": self.dueling,
            "noisy": self.noisy,
            "rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() and self.device.type == "cuda"
                else None
            ),
            "buffer": self.buffer.state_dict(),
        }

    def _restore_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        if not metadata:
            return
        action_dim = int(metadata.get("action_dim", self.action_dim))
        if action_dim != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match the current agent.")
        use_per = bool(metadata.get("use_per", self.use_per))
        if use_per != self.use_per:
            raise ValueError("Checkpoint replay strategy does not match the current agent.")
        if bool(metadata.get("dueling", self.dueling)) != self.dueling:
            raise ValueError("Checkpoint dueling configuration does not match the current agent.")
        if bool(metadata.get("noisy", self.noisy)) != self.noisy:
            raise ValueError("Checkpoint noisy-network configuration does not match the current agent.")
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

    def _soft_update(self) -> None:
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target_param.data.mul_(1.0 - self.tau).add_(online_param.data, alpha=self.tau)
