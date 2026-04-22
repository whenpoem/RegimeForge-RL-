"""Regime-Conditioned Mixture-of-Experts DQN (RCMoE-DQN).

This module implements the core research contribution: a DQN agent whose
Q-network is decomposed into *N* expert sub-networks plus a gating
network that implicitly infers the current market regime from
observations and routes decision-making to the appropriate expert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import GateType
from .dqn import (
    NoisyLinear,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    load_agent_checkpoint_bundle,
    save_agent_checkpoint_bundle,
)


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

class MLPGatingNetwork(nn.Module):
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


class TemporalGatingNetwork(nn.Module):
    """Attention-based gate over a short observation history."""

    def __init__(self, observation_dim: int, n_experts: int, hidden_dim: int = 64, context_len: int = 8):
        super().__init__()
        self.observation_dim = observation_dim
        self.context_len = context_len
        self.embed = nn.Linear(observation_dim, hidden_dim)
        n_heads = 4 if hidden_dim % 4 == 0 else 1
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.view(x.size(0), self.context_len, self.observation_dim)
        embedded = self.embed(seq)
        attended, _ = self.attention(embedded, embedded, embedded, need_weights=False)
        pooled = attended[:, -1, :]
        return torch.softmax(self.proj(pooled), dim=-1)


class HierarchicalGatingNetwork(nn.Module):
    """Two-level gate: macro allocation then local expert routing."""

    def __init__(self, observation_dim: int, n_experts: int, hidden_dim: int = 64, macro_experts: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.macro_experts = max(1, min(macro_experts, n_experts))
        group_sizes = [n_experts // self.macro_experts for _ in range(self.macro_experts)]
        for index in range(n_experts % self.macro_experts):
            group_sizes[index] += 1
        self.group_sizes = group_sizes
        self.macro_gate = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.macro_experts),
        )
        self.local_gates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, group_size),
            )
            for group_size in group_sizes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        macro = torch.softmax(self.macro_gate(x), dim=-1)
        parts: list[torch.Tensor] = []
        for group_idx, local_gate in enumerate(self.local_gates):
            local = torch.softmax(local_gate(x), dim=-1)
            parts.append(local * macro[:, group_idx : group_idx + 1])
        return torch.cat(parts, dim=-1)


class ExpertNetwork(nn.Module):
    """A single expert Q-network specialising on one regime."""

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
        self.trunk = nn.Sequential(
            linear(observation_dim, hidden_dim),
            nn.SiLU(),
            linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Backwards-compatible alias used by older tests and callers.
        self.layers = self.trunk
        if dueling:
            self.value_head = linear(hidden_dim, 1)
            self.advantage_head = linear(hidden_dim, action_dim)
            self.output_head = None
        else:
            self.value_head = None
            self.advantage_head = None
            self.output_head = linear(hidden_dim, action_dim)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(x)
        if self.dueling:
            assert self.value_head is not None
            assert self.advantage_head is not None
            value = self.value_head(hidden)
            advantage = self.advantage_head(hidden)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        assert self.output_head is not None
        return self.output_head(hidden)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RCMoEQNetwork(nn.Module):
    """Mixture-of-Experts Q-network with gated regime routing.

    The mixed Q-value is:
        Q̂(s, a) = Σᵢ wᵢ(s) · Qᵢ(s, a)

    where wᵢ(s) are soft gate weights and Qᵢ are expert Q-values.
    """

    def __init__(
        self,
        input_dim: int,
        base_observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_experts: int = 4,
        gate_hidden_dim: int = 64,
        gate_type: GateType = GateType.MLP,
        context_len: int = 8,
        hierarchical_moe: bool = False,
        macro_experts: int = 2,
        dueling: bool = False,
        noisy: bool = False,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.base_observation_dim = base_observation_dim
        self.context_len = context_len
        self.gate_type = gate_type
        if hierarchical_moe:
            self.gate = HierarchicalGatingNetwork(base_observation_dim, n_experts, gate_hidden_dim, macro_experts)
        elif gate_type == GateType.TEMPORAL:
            self.gate = TemporalGatingNetwork(base_observation_dim, n_experts, gate_hidden_dim, context_len)
        else:
            self.gate = MLPGatingNetwork(base_observation_dim, n_experts, gate_hidden_dim)
        self.experts = nn.ModuleList([
            ExpertNetwork(base_observation_dim, action_dim, hidden_dim, dueling=dueling, noisy=noisy)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mixed_q, gate_weights).

        mixed_q:      (B, action_dim)
        gate_weights:  (B, n_experts)
        """
        latest_state = self._latest_state(x)
        weights = self._gate_weights(x, latest_state)                                  # (B, n_experts)
        expert_qs = torch.stack([expert(latest_state) for expert in self.experts], dim=1)
        mixed_q = (weights.unsqueeze(-1) * expert_qs).sum(dim=1)       # (B, action_dim)
        return mixed_q, weights

    def expert_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw Q-values from every expert: (B, n_experts, action_dim)."""
        latest_state = self._latest_state(x)
        return torch.stack([expert(latest_state) for expert in self.experts], dim=1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        latest_state = self._latest_state(x)
        weights = self._gate_weights(x, latest_state)
        expert_hidden = torch.stack([expert.hidden(latest_state) for expert in self.experts], dim=1)
        return (weights.unsqueeze(-1) * expert_hidden).sum(dim=1)

    def reset_noise(self) -> None:
        for expert in self.experts:
            expert.reset_noise()

    def _latest_state(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_type == GateType.TEMPORAL:
            seq = x.view(x.size(0), self.context_len, self.base_observation_dim)
            return seq[:, -1, :]
        return x

    def _gate_weights(self, x: torch.Tensor, latest_state: torch.Tensor) -> torch.Tensor:
        if self.gate_type == GateType.TEMPORAL:
            return self.gate(x)
        return self.gate(latest_state)


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
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_epsilon: float = 1e-6,
        gate_type: GateType = GateType.MLP,
        context_len: int = 8,
        hierarchical_moe: bool = False,
        macro_experts: int = 2,
        dueling: bool = False,
        noisy: bool = False,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.load_balance_weight = load_balance_weight
        self.use_per = use_per
        self.gate_type = gate_type
        self.context_len = context_len
        self.noisy = noisy
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.online = RCMoEQNetwork(
            observation_dim,
            observation_dim if gate_type != GateType.TEMPORAL else observation_dim // context_len,
            action_dim,
            hidden_dim,
            n_experts,
            gate_hidden_dim,
            gate_type=gate_type,
            context_len=context_len,
            hierarchical_moe=hierarchical_moe,
            macro_experts=macro_experts,
            dueling=dueling,
            noisy=noisy,
        ).to(self.device)
        self.target = RCMoEQNetwork(
            observation_dim,
            observation_dim if gate_type != GateType.TEMPORAL else observation_dim // context_len,
            action_dim,
            hidden_dim,
            n_experts,
            gate_hidden_dim,
            gate_type=gate_type,
            context_len=context_len,
            hierarchical_moe=hierarchical_moe,
            macro_experts=macro_experts,
            dueling=dueling,
            noisy=noisy,
        ).to(self.device)
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
        self.n_experts = n_experts

        # Running gate-weight statistics (updated during training)
        self._last_gate_weights: np.ndarray | None = None

    # -- Action selection ----------------------------------------------------

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if self.noisy:
            self.online.reset_noise()
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

    def update(self, per_beta: float = 0.4) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        if self.use_per:
            states_np, actions_np, rewards_np, next_np, dones_np, is_weights_np, tree_indices = \
                self.buffer.sample(self.batch_size, self.rng, beta=per_beta)
            is_weights = torch.from_numpy(is_weights_np).to(self.device, non_blocking=True)
        else:
            states_np, actions_np, rewards_np, next_np, dones_np = \
                self.buffer.sample(self.batch_size, self.rng)
            is_weights = None

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
            if self.noisy:
                self.online.reset_noise()
                self.target.reset_noise()
            next_q_online, _ = self.online(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target, _ = self.target(next_states)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        td_errors = self.loss_fn(q_selected, target_q)  # per-sample
        if is_weights is not None:
            td_loss = (td_errors * is_weights).mean()
        else:
            td_loss = td_errors.mean()

        # Load-balancing auxiliary loss
        lb_loss = load_balance_loss(gate_w)
        total_loss = td_loss + self.load_balance_weight * lb_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._soft_update()

        # Update PER priorities with raw TD errors
        if self.use_per:
            with torch.no_grad():
                raw_td = (q_selected - target_q).abs().cpu().numpy()
            self.buffer.update_priorities(tree_indices, raw_td)

        return float(td_loss.item())

    def hidden_activations(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            hidden = self.online.hidden(state_t)
        return hidden.squeeze(0).cpu().numpy().astype(np.float64)

    def batch_hidden_activations(self, states: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            hidden = self.online.hidden(state_t)
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
            "n_experts": self.n_experts,
            "use_per": self.use_per,
            "gate_type": self.gate_type.value,
            "context_len": self.context_len,
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
        n_experts = int(metadata.get("n_experts", self.n_experts))
        if n_experts != self.n_experts:
            raise ValueError("Checkpoint expert count does not match the current agent.")
        use_per = bool(metadata.get("use_per", self.use_per))
        if use_per != self.use_per:
            raise ValueError("Checkpoint replay strategy does not match the current agent.")
        gate_type = str(metadata.get("gate_type", self.gate_type.value))
        if gate_type != self.gate_type.value:
            raise ValueError("Checkpoint gate type does not match the current agent.")
        if int(metadata.get("context_len", self.context_len)) != self.context_len:
            raise ValueError("Checkpoint context length does not match the current agent.")
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

    # -- Target-network soft update ------------------------------------------

    def _soft_update(self) -> None:
        for tp, op in zip(self.target.parameters(), self.online.parameters(), strict=True):
            tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)
