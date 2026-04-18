from __future__ import annotations

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


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


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
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.online = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(replay_capacity, observation_dim)
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
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

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        states_np, actions_np, rewards_np, next_states_np, dones_np = self.buffer.sample(self.batch_size, self.rng)
        states = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions_np).to(self.device, non_blocking=True).unsqueeze(1)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)

        q_selected = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_selected, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._soft_update()
        return float(loss.item())

    def hidden_activations(self, state: np.ndarray) -> np.ndarray:
        """Extract the hidden-layer activations before the output head.

        Returns the output of the second hidden layer (after SiLU),
        useful for linear-probing regime analysis.
        Shape: (hidden_dim,)
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        activation: list[torch.Tensor] = []

        def _hook(module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
            activation.append(output.detach())

        # Register hook on the 4th layer (second SiLU, index 3)
        handle = self.online.layers[3].register_forward_hook(_hook)
        with torch.inference_mode():
            self.online(state_tensor)
        handle.remove()

        if activation:
            return activation[0].squeeze(0).cpu().numpy().astype(np.float64)
        return np.zeros(self.online.layers[2].out_features, dtype=np.float64)

    def batch_hidden_activations(self, states: np.ndarray) -> np.ndarray:
        """Extract hidden activations for a batch of states.

        Returns shape (B, hidden_dim).
        """
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        activation: list[torch.Tensor] = []

        def _hook(module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
            activation.append(output.detach())

        handle = self.online.layers[3].register_forward_hook(_hook)
        with torch.inference_mode():
            self.online(state_tensor)
        handle.remove()

        if activation:
            return activation[0].cpu().numpy().astype(np.float64)
        return np.zeros((len(states), self.online.layers[2].out_features), dtype=np.float64)

    def _soft_update(self) -> None:
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target_param.data.mul_(1.0 - self.tau).add_(online_param.data, alpha=self.tau)
