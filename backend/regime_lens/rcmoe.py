"""Regime-Conditioned Mixture-of-Experts DQN (RCMoE-DQN).

This module implements the core research contribution: a DQN agent whose
Q-network is decomposed into *N* expert sub-networks plus a gating
network that implicitly infers the current market regime from
observations and routes decision-making to the appropriate expert.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Soft gating network that infers regime from state.

    Outputs a probability distribution over *n_experts* experts,
    interpretable as an implicit regime posterior.
    """

    def __init__(self, observation_dim: int, n_experts: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, n_experts) soft weights summing to 1."""
        return torch.softmax(self.net(x), dim=-1)


class ExpertNetwork(nn.Module):
    """A single expert Q-network specialising on one regime."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RCMoEQNetwork(nn.Module):
    """Mixture-of-Experts Q-network with gated regime routing.

    The mixed Q-value is:
        Q̂(s, a) = Σᵢ wᵢ(s) · Qᵢ(s, a)

    where wᵢ(s) are soft gate weights and Qᵢ are expert Q-values.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_experts: int = 4,
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.gate = GatingNetwork(observation_dim, n_experts, gate_hidden_dim)
        self.experts = nn.ModuleList([
            ExpertNetwork(observation_dim, action_dim, hidden_dim)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mixed_q, gate_weights).

        mixed_q:      (B, action_dim)
        gate_weights:  (B, n_experts)
        """
        weights = self.gate(x)                                         # (B, n_experts)
        expert_qs = torch.stack([e(x) for e in self.experts], dim=1)   # (B, n_experts, action_dim)
        mixed_q = (weights.unsqueeze(-1) * expert_qs).sum(dim=1)       # (B, action_dim)
        return mixed_q, weights

    def expert_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw Q-values from every expert: (B, n_experts, action_dim)."""
        return torch.stack([e(x) for e in self.experts], dim=1)


# ---------------------------------------------------------------------------
# Load-balancing auxiliary loss
# ---------------------------------------------------------------------------

def load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Auxiliary loss encouraging uniform expert utilisation.

    Implements the importance-weighted load-balancing loss from
    `Switch Transformers (Fedus et al., 2022)`:

        L_lb = n_experts · Σᵢ fᵢ · Pᵢ

    where fᵢ is the fraction of tokens routed to expert i and Pᵢ is
    the average gate probability for expert i.
    """
    # gate_weights: (B, n_experts)
    n_experts = gate_weights.size(1)
    # Fraction dispatched (hard routing proxy)
    hard_assignments = gate_weights.argmax(dim=-1)                     # (B,)
    f_i = torch.zeros(n_experts, device=gate_weights.device)
    for i in range(n_experts):
        f_i[i] = (hard_assignments == i).float().mean()
    # Average gate probability per expert
    p_i = gate_weights.mean(dim=0)                                     # (n_experts,)
    return n_experts * (f_i * p_i).sum()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RCMoEAgent:
    """RCMoE-DQN agent with Double-DQN updates + load-balancing loss."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_experts: int,
        gate_hidden_dim: int,
        load_balance_weight: float,
        learning_rate: float,
        gamma: float,
        tau: float,
        replay_capacity: int,
        batch_size: int,
        device: str,
        seed: int,
    ):
        from .dqn import ReplayBuffer

        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.load_balance_weight = load_balance_weight
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.online = RCMoEQNetwork(
            observation_dim, action_dim, hidden_dim, n_experts, gate_hidden_dim,
        ).to(self.device)
        self.target = RCMoEQNetwork(
            observation_dim, action_dim, hidden_dim, n_experts, gate_hidden_dim,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(replay_capacity, observation_dim)
        self.action_dim = action_dim
        self.n_experts = n_experts

        # Running gate-weight statistics (updated during training)
        self._last_gate_weights: np.ndarray | None = None

    # -- Action selection ----------------------------------------------------

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_dim))
        return self.greedy_action(state)

    def greedy_action(self, state: np.ndarray) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            q, weights = self.online(state_t)
        self._last_gate_weights = weights.squeeze(0).cpu().numpy()
        return int(torch.argmax(q, dim=1).item())

    # -- Q-value access ------------------------------------------------------

    def q_values(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            q, _ = self.online(state_t)
        return q.squeeze(0).cpu().numpy().astype(np.float64)

    def batch_q_values(self, states: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            q, _ = self.online(state_t)
        return q.cpu().numpy().astype(np.float64, copy=False)

    # -- Gate-weight access --------------------------------------------------

    def gate_weights(self, state: np.ndarray) -> np.ndarray:
        """Return gate weights for a single observation: shape (n_experts,)."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            _, w = self.online(state_t)
        return w.squeeze(0).cpu().numpy().astype(np.float64)

    def batch_gate_weights(self, states: np.ndarray) -> np.ndarray:
        """Return gate weights for a batch: shape (B, n_experts)."""
        state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            _, w = self.online(state_t)
        return w.cpu().numpy().astype(np.float64, copy=False)

    @property
    def last_gate_weights(self) -> np.ndarray | None:
        """Most recent gate weights from ``greedy_action``."""
        return self._last_gate_weights

    # -- Experience storage --------------------------------------------------

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> None:
        self.buffer.add(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

    # -- Learning step -------------------------------------------------------

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states_np, actions_np, rewards_np, next_np, dones_np = \
            self.buffer.sample(self.batch_size, self.rng)

        states = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions_np).to(self.device, non_blocking=True).unsqueeze(1)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)

        # Online Q for current states
        q_all, gate_w = self.online(states)
        q_selected = q_all.gather(1, actions).squeeze(1)

        # Double-DQN target
        with torch.no_grad():
            next_q_online, _ = self.online(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target, _ = self.target(next_states)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        td_loss = self.loss_fn(q_selected, target_q)

        # Load-balancing auxiliary loss
        lb_loss = load_balance_loss(gate_w)
        total_loss = td_loss + self.load_balance_weight * lb_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._soft_update()
        return float(td_loss.item())

    # -- Target-network soft update ------------------------------------------

    def _soft_update(self) -> None:
        for tp, op in zip(self.target.parameters(), self.online.parameters(), strict=True):
            tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)
