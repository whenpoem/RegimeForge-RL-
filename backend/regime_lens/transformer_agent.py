"""Transformer-based sequence policy for RegimeForge.

Uses a Transformer encoder over a sliding window of observations to
capture temporal dependencies that MLP-based agents miss.  Supports
both discrete (DQN-style) and continuous (PPO/SAC-style) action spaces.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque

import numpy as np
import torch
from torch import nn

from .config import AgentType, AlgorithmType, TrainingConfig, config_to_snapshot
from .dqn import PrioritizedReplayBuffer, ReplayBuffer, SumTree
from .context import build_temporal_context, initialise_context_history, append_context_state


# ---------------------------------------------------------------------------
# Transformer Q-Network (discrete actions)
# ---------------------------------------------------------------------------

class TransformerQNetwork(nn.Module):
    """Q-network with Transformer encoder over observation sequences."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.seq_len = seq_len
        self.input_proj = nn.Linear(observation_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len * obs_dim) flattened context -> (B, action_dim)"""
        seq = x.view(x.size(0), self.seq_len, self.observation_dim)
        h = self.input_proj(seq) + self.pos_embed
        h = self.encoder(h)
        pooled = h[:, -1, :]  # last token
        return self.head(pooled)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.view(x.size(0), self.seq_len, self.observation_dim)
        h = self.input_proj(seq) + self.pos_embed
        h = self.encoder(h)
        return h[:, -1, :]


# ---------------------------------------------------------------------------
# Transformer Discrete Agent
# ---------------------------------------------------------------------------

class TransformerDQNAgent:
    """DQN agent using a Transformer encoder over observation history."""

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
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 16,
        dropout: float = 0.1,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_epsilon: float = 1e-6,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_per = use_per
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.online = TransformerQNetwork(
            observation_dim, action_dim, hidden_dim, n_heads, n_layers, seq_len, dropout,
        ).to(self.device)
        self.target = TransformerQNetwork(
            observation_dim, action_dim, hidden_dim, n_heads, n_layers, seq_len, dropout,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                replay_capacity, observation_dim * seq_len, alpha=per_alpha, epsilon=per_epsilon,
            )
        else:
            self.buffer = ReplayBuffer(replay_capacity, observation_dim * seq_len)
        self._context_history: Deque[np.ndarray] | None = None
        self._last_state: np.ndarray | None = None
        self._last_context: np.ndarray | None = None

    def reset_context(self) -> None:
        self._context_history = None
        self._last_state = None
        self._last_context = None

    def _coerce_state(self, state: np.ndarray) -> np.ndarray:
        array = np.asarray(state, dtype=np.float32).reshape(-1)
        if array.shape[0] != self.observation_dim:
            raise ValueError("State shape does not match the configured observation_dim.")
        return array

    def _same_state(self, state: np.ndarray) -> bool:
        return self._last_state is not None and self._last_state.shape == state.shape and np.array_equal(self._last_state, state)

    def _build_context(self, state: np.ndarray, *, advance: bool = False) -> np.ndarray:
        state_array = self._coerce_state(state)
        if self._context_history is None:
            history = initialise_context_history(state_array, context_len=self.seq_len)
        elif advance:
            history = append_context_state(self._context_history, state_array)
        elif self._same_state(state_array) and self._last_context is not None:
            return self._last_context.copy()
        else:
            history = append_context_state(self._context_history, state_array)

        context = build_temporal_context(history, context_len=self.seq_len)
        if advance:
            self._context_history = history
            self._last_state = state_array.copy()
            self._last_context = context.copy()
        return context

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        ctx = self._build_context(state, advance=True)
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_dim))
        return self._greedy_from_context(ctx)

    def greedy_action(self, state: np.ndarray) -> int:
        ctx = self._build_context(state, advance=True)
        return self._greedy_from_context(ctx)

    def _greedy_from_context(self, context: np.ndarray) -> int:
        tensor = torch.as_tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            q = self.online(tensor)
        return int(torch.argmax(q, dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        ctx = self._build_context(state, advance=False)
        tensor = torch.as_tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            q = self.online(tensor)
        return q.squeeze(0).cpu().numpy().astype(np.float64)

    def batch_q_values(self, states: np.ndarray) -> np.ndarray:
        states_array = np.asarray(states, dtype=np.float32)
        if states_array.ndim != 2 or states_array.shape[1] != self.observation_dim * self.seq_len:
            raise ValueError("Transformer batch_q_values expects flattened temporal contexts.")
        tensor = torch.as_tensor(states_array, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            q = self.online(tensor)
        return q.cpu().numpy().astype(np.float64, copy=False)

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> None:
        ctx = self._build_context(state, advance=False)
        next_ctx = self._build_context(next_state, advance=False)
        self.buffer.add(
            np.asarray(ctx, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_ctx, dtype=np.float32),
            bool(done),
        )

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

        q_selected = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        td_errors = self.loss_fn(q_selected, target_q)
        if is_weights is not None:
            loss = (td_errors * is_weights).mean()
        else:
            loss = td_errors.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._soft_update()

        if self.use_per:
            with torch.no_grad():
                raw_td = (q_selected - target_q).abs().cpu().numpy()
            self.buffer.update_priorities(tree_indices, raw_td)
        return float(loss.item())

    def hidden_activations(self, state: np.ndarray) -> np.ndarray:
        ctx = self._build_context(state, advance=False)
        tensor = torch.as_tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            hidden = self.online.hidden(tensor)
        return hidden.squeeze(0).cpu().numpy().astype(np.float64)

    def batch_hidden_activations(self, states: np.ndarray) -> np.ndarray:
        states_array = np.asarray(states, dtype=np.float32)
        if states_array.ndim != 2 or states_array.shape[1] != self.observation_dim * self.seq_len:
            raise ValueError("Transformer batch_hidden_activations expects flattened temporal contexts.")
        tensor = torch.as_tensor(states_array, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            hidden = self.online.hidden(tensor)
        return hidden.cpu().numpy().astype(np.float64, copy=False)

    def save_checkpoint(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.online.state_dict(), d / "online.pt")
        torch.save(self.target.state_dict(), d / "target.pt")
        torch.save(self.optimizer.state_dict(), d / "optimizer.pt")
        torch.save({
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "seq_len": self.seq_len,
            "rng_state": self.rng.bit_generator.state,
            "buffer": self.buffer.state_dict(),
        }, d / "meta.pt")

    def load_checkpoint(self, directory: str | Path, weights_only: bool = True) -> None:
        d = Path(directory)
        self.online.load_state_dict(torch.load(d / "online.pt", map_location=self.device, weights_only=True))
        self.target.load_state_dict(torch.load(d / "target.pt", map_location=self.device, weights_only=True))
        self.target.eval()
        if not weights_only:
            opt_path = d / "optimizer.pt"
            if opt_path.exists():
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device, weights_only=True))
            meta_path = d / "meta.pt"
            if meta_path.exists():
                meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                if isinstance(meta, dict):
                    buffer_state = meta.get("buffer")
                    if isinstance(buffer_state, dict):
                        self.buffer.load_state_dict(buffer_state)
                    rng_state = meta.get("rng_state")
                    if rng_state is not None:
                        self.rng.bit_generator.state = rng_state

    def _soft_update(self) -> None:
        for tp, op in zip(self.target.parameters(), self.online.parameters(), strict=True):
            tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)


__all__ = [
    "TransformerDQNAgent",
    "TransformerQNetwork",
]
