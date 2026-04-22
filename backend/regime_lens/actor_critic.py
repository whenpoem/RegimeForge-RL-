from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .config import AgentType, AlgorithmType, TrainingConfig


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass(slots=True)
class PolicySample:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    extras: dict[str, torch.Tensor]


def _resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cpu")
    return torch.device(device)


def _ensure_tensor(value: np.ndarray | torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, bool]:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    squeezed = tensor.ndim == 1
    if squeezed:
        tensor = tensor.unsqueeze(0)
    return tensor, squeezed


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class GaussianPolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value(hidden)


class QHead(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observation, action], dim=-1))


class PPOActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.encoder = MLPEncoder(observation_dim, hidden_dim)
        self.actor = GaussianPolicyHead(hidden_dim, action_dim)
        self.critic = ValueHead(hidden_dim)

    def forward(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(observation)
        mean, log_std = self.actor(hidden)
        value = self.critic(hidden)
        return {"mean": mean, "log_std": log_std, "value": value}

    def sample_action(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> PolicySample:
        outputs = self.forward(observation)
        distribution = Normal(outputs["mean"], outputs["log_std"].exp())
        pre_tanh = outputs["mean"] if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        correction = torch.log1p(-action.pow(2) + 1e-6)
        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        extras = dict(outputs)
        extras["pre_tanh_action"] = pre_tanh
        return PolicySample(action=action, log_prob=log_prob, entropy=entropy, extras=extras)

    def act(
        self, observation: np.ndarray | torch.Tensor, *, deterministic: bool = False
    ) -> dict[str, np.ndarray | torch.Tensor]:
        device = next(self.parameters()).device
        tensor, squeezed = _ensure_tensor(observation, device=device)
        sample = self.sample_action(tensor, deterministic=deterministic)
        action = sample.action.squeeze(0) if squeezed else sample.action
        value = sample.extras["value"].squeeze(0) if squeezed else sample.extras["value"]
        log_prob = sample.log_prob.squeeze(0) if squeezed else sample.log_prob
        return {"action": action, "value": value, "log_prob": log_prob}

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward(observation)
        eps = 1e-6
        clipped_action = action.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(clipped_action) - torch.log1p(-clipped_action))
        distribution = Normal(outputs["mean"], outputs["log_std"].exp())
        correction = torch.log1p(-clipped_action.pow(2) + eps)
        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        return {"log_prob": log_prob, "entropy": entropy, "value": outputs["value"]}


class SACActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.encoder = MLPEncoder(observation_dim, hidden_dim)
        self.actor = GaussianPolicyHead(hidden_dim, action_dim)
        self.q1 = QHead(observation_dim, action_dim, hidden_dim)
        self.q2 = QHead(observation_dim, action_dim, hidden_dim)

    def forward(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(observation)
        mean, log_std = self.actor(hidden)
        return {"mean": mean, "log_std": log_std}

    def sample_action(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> PolicySample:
        outputs = self.forward(observation)
        distribution = Normal(outputs["mean"], outputs["log_std"].exp())
        pre_tanh = outputs["mean"] if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        correction = torch.log1p(-action.pow(2) + 1e-6)
        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        extras = dict(outputs)
        extras["pre_tanh_action"] = pre_tanh
        return PolicySample(action=action, log_prob=log_prob, entropy=entropy, extras=extras)

    def act(
        self, observation: np.ndarray | torch.Tensor, *, deterministic: bool = False
    ) -> dict[str, np.ndarray | torch.Tensor]:
        device = next(self.parameters()).device
        tensor, squeezed = _ensure_tensor(observation, device=device)
        sample = self.sample_action(tensor, deterministic=deterministic)
        action = sample.action.squeeze(0) if squeezed else sample.action
        log_prob = sample.log_prob.squeeze(0) if squeezed else sample.log_prob
        return {"action": action, "log_prob": log_prob}

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward(observation)
        eps = 1e-6
        clipped_action = action.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(clipped_action) - torch.log1p(-clipped_action))
        distribution = Normal(outputs["mean"], outputs["log_std"].exp())
        correction = torch.log1p(-clipped_action.pow(2) + eps)
        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        return {
            "log_prob": log_prob,
            "q1": self.q1(observation, clipped_action),
            "q2": self.q2(observation, clipped_action),
        }

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.actor.parameters())

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self.q1.parameters()) + list(self.q2.parameters())

    def critic_freeze_parameters(self) -> list[nn.Parameter]:
        return self.critic_parameters()


class _MixtureGate(nn.Module):
    def __init__(self, observation_dim: int, n_experts: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(observation), dim=-1)


class _ExpertGaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = MLPEncoder(observation_dim, hidden_dim)
        self.head = GaussianPolicyHead(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.encoder(observation))


class _ExpertValueCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = MLPEncoder(observation_dim, hidden_dim)
        self.head = ValueHead(hidden_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(observation))


class RCMoEActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_experts: int = 4,
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_experts = n_experts
        self.gate = _MixtureGate(observation_dim, n_experts, gate_hidden_dim)
        self.expert_actors = nn.ModuleList(
            _ExpertGaussianActor(observation_dim, action_dim, hidden_dim) for _ in range(n_experts)
        )
        self.expert_critics = nn.ModuleList(
            _ExpertValueCritic(observation_dim, hidden_dim) for _ in range(n_experts)
        )

    def expert_outputs(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        gate_weights = self.gate(observation)
        expert_means: list[torch.Tensor] = []
        expert_log_stds: list[torch.Tensor] = []
        expert_values: list[torch.Tensor] = []

        for actor, critic in zip(self.expert_actors, self.expert_critics, strict=True):
            mean, log_std = actor(observation)
            expert_means.append(mean)
            expert_log_stds.append(log_std)
            expert_values.append(critic(observation))

        return {
            "gate_weights": gate_weights,
            "expert_means": torch.stack(expert_means, dim=1),
            "expert_log_stds": torch.stack(expert_log_stds, dim=1),
            "expert_values": torch.stack(expert_values, dim=1).squeeze(-1),
        }

    def forward(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.expert_outputs(observation)
        gate = outputs["gate_weights"]
        mean = (gate.unsqueeze(-1) * outputs["expert_means"]).sum(dim=1)
        log_std = (gate.unsqueeze(-1) * outputs["expert_log_stds"]).sum(dim=1).clamp(LOG_STD_MIN, LOG_STD_MAX)
        value = (gate * outputs["expert_values"]).sum(dim=1, keepdim=True)
        outputs["mean"] = mean
        outputs["log_std"] = log_std
        outputs["value"] = value
        return outputs

    def sample_action(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> PolicySample:
        outputs = self.forward(observation)
        distribution = Normal(outputs["mean"], outputs["log_std"].exp())
        pre_tanh = outputs["mean"] if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        correction = torch.log1p(-action.pow(2) + 1e-6)
        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        extras = dict(outputs)
        extras["pre_tanh_action"] = pre_tanh
        return PolicySample(action=action, log_prob=log_prob, entropy=entropy, extras=extras)

    def act(
        self, observation: np.ndarray | torch.Tensor, *, deterministic: bool = False
    ) -> dict[str, np.ndarray | torch.Tensor]:
        device = next(self.parameters()).device
        tensor, squeezed = _ensure_tensor(observation, device=device)
        sample = self.sample_action(tensor, deterministic=deterministic)
        action = sample.action.squeeze(0) if squeezed else sample.action
        gate = sample.extras["gate_weights"].squeeze(0) if squeezed else sample.extras["gate_weights"]
        value = sample.extras["value"].squeeze(0) if squeezed else sample.extras["value"]
        return {"action": action, "gate_weights": gate, "value": value, "log_prob": sample.log_prob}

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward(observation)
        eps = 1e-6
        clipped_action = action.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(clipped_action) - torch.log1p(-clipped_action))
        distribution = Normal(outputs["mean"], outputs["log_std"].exp())
        correction = torch.log1p(-clipped_action.pow(2) + eps)
        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": outputs["value"],
            "gate_weights": outputs["gate_weights"],
        }


class RCMoESACActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_experts: int = 4,
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_experts = n_experts
        self.gate = _MixtureGate(observation_dim, n_experts, gate_hidden_dim)
        self.expert_actors = nn.ModuleList(
            _ExpertGaussianActor(observation_dim, action_dim, hidden_dim) for _ in range(n_experts)
        )
        self.expert_q1 = nn.ModuleList(
            QHead(observation_dim, action_dim, hidden_dim) for _ in range(n_experts)
        )
        self.expert_q2 = nn.ModuleList(
            QHead(observation_dim, action_dim, hidden_dim) for _ in range(n_experts)
        )

    def expert_outputs(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        gate_weights = self.gate(observation)
        expert_means: list[torch.Tensor] = []
        expert_log_stds: list[torch.Tensor] = []

        for actor in self.expert_actors:
            mean, log_std = actor(observation)
            expert_means.append(mean)
            expert_log_stds.append(log_std)

        return {
            "gate_weights": gate_weights,
            "expert_means": torch.stack(expert_means, dim=1),
            "expert_log_stds": torch.stack(expert_log_stds, dim=1),
        }

    def forward(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.expert_outputs(observation)
        gate = outputs["gate_weights"]
        outputs["mean"] = (gate.unsqueeze(-1) * outputs["expert_means"]).sum(dim=1)
        outputs["log_std"] = (
            gate.unsqueeze(-1) * outputs["expert_log_stds"]
        ).sum(dim=1).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return outputs

    def _critic_value(
        self,
        critics: nn.ModuleList,
        observation: torch.Tensor,
        action: torch.Tensor,
        gate_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = self.gate(observation) if gate_weights is None else gate_weights
        expert_values = torch.stack(
            [critic(observation, action) for critic in critics],
            dim=1,
        ).squeeze(-1)
        return (gate * expert_values).sum(dim=1, keepdim=True)

    def q1(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._critic_value(self.expert_q1, observation, action)

    def q2(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._critic_value(self.expert_q2, observation, action)

    def sample_action(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> PolicySample:
        outputs = self.forward(observation)
        distribution = Normal(outputs["mean"], outputs["log_std"].exp())
        pre_tanh = outputs["mean"] if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        correction = torch.log1p(-action.pow(2) + 1e-6)
        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        extras = dict(outputs)
        extras["pre_tanh_action"] = pre_tanh
        return PolicySample(action=action, log_prob=log_prob, entropy=entropy, extras=extras)

    def act(
        self, observation: np.ndarray | torch.Tensor, *, deterministic: bool = False
    ) -> dict[str, np.ndarray | torch.Tensor]:
        device = next(self.parameters()).device
        tensor, squeezed = _ensure_tensor(observation, device=device)
        sample = self.sample_action(tensor, deterministic=deterministic)
        action = sample.action.squeeze(0) if squeezed else sample.action
        gate = sample.extras["gate_weights"].squeeze(0) if squeezed else sample.extras["gate_weights"]
        log_prob = sample.log_prob.squeeze(0) if squeezed else sample.log_prob
        return {"action": action, "gate_weights": gate, "log_prob": log_prob}

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward(observation)
        eps = 1e-6
        clipped_action = action.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(clipped_action) - torch.log1p(-clipped_action))
        distribution = Normal(outputs["mean"], outputs["log_std"].exp())
        correction = torch.log1p(-clipped_action.pow(2) + eps)
        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True) - correction.sum(dim=-1, keepdim=True)
        gate_weights = outputs["gate_weights"]
        return {
            "log_prob": log_prob,
            "q1": self._critic_value(self.expert_q1, observation, clipped_action, gate_weights),
            "q2": self._critic_value(self.expert_q2, observation, clipped_action, gate_weights),
            "gate_weights": gate_weights,
        }

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.gate.parameters()) + list(self.expert_actors.parameters())

    def critic_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.gate.parameters())
            + list(self.expert_q1.parameters())
            + list(self.expert_q2.parameters())
        )

    def critic_freeze_parameters(self) -> list[nn.Parameter]:
        return list(self.expert_q1.parameters()) + list(self.expert_q2.parameters())


def build_actor_critic(
    config: TrainingConfig,
    *,
    observation_dim: int,
    action_dim: int,
    device: str | None = None,
    variant: str | None = None,
) -> nn.Module:
    resolved_device = _resolve_device(device or config.device)
    model_kind = (variant or "").strip().lower()
    if not model_kind:
        if config.agent_type == AgentType.RCMOE_DQN:
            model_kind = "rcmoe"
        elif config.algorithm == AlgorithmType.SAC:
            model_kind = "sac"
        else:
            model_kind = "ppo"

    if model_kind == "ppo":
        model: nn.Module = PPOActorCritic(observation_dim, action_dim, hidden_dim=config.hidden_dim)
    elif model_kind == "sac":
        model = SACActorCritic(observation_dim, action_dim, hidden_dim=config.hidden_dim)
    elif model_kind in {"rcmoe", "rcmoe_actor_critic", "moe", "rcmoe_sac", "moe_sac"}:
        if config.algorithm == AlgorithmType.SAC or model_kind in {"rcmoe_sac", "moe_sac"}:
            model = RCMoESACActorCritic(
                observation_dim,
                action_dim,
                hidden_dim=config.hidden_dim,
                n_experts=config.n_experts,
                gate_hidden_dim=config.gate_hidden_dim,
            )
        else:
            model = RCMoEActorCritic(
                observation_dim,
                action_dim,
                hidden_dim=config.hidden_dim,
                n_experts=config.n_experts,
                gate_hidden_dim=config.gate_hidden_dim,
            )
    else:
        raise ValueError(f"Unsupported actor-critic variant: {model_kind}")

    return model.to(resolved_device)


__all__ = [
    "PPOActorCritic",
    "PolicySample",
    "RCMoEActorCritic",
    "RCMoESACActorCritic",
    "SACActorCritic",
    "build_actor_critic",
]
