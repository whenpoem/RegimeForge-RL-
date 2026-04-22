"""Regime Lens backend package."""

from .config import AgentType, CurriculumMode, TrainingConfig
from .continuous_market import ContinuousMarketEnv, MultiAssetContinuousMarketEnv, make_market_env
from .actor_critic import PPOActorCritic, RCMoEActorCritic, SACActorCritic, build_actor_critic
from .continuous_agent import ContinuousActorCriticAgent
from .explainability import (
    decision_boundary,
    expert_counterfactual,
    find_transition_points,
    gate_attribution,
    transition_lag,
)
from .training import TrainingManager
from .runtime import RuntimeInfo

__all__ = [
    "AgentType",
    "ContinuousMarketEnv",
    "ContinuousActorCriticAgent",
    "CurriculumMode",
    "MultiAssetContinuousMarketEnv",
    "PPOActorCritic",
    "RCMoEActorCritic",
    "RuntimeInfo",
    "SACActorCritic",
    "TrainingConfig",
    "TrainingManager",
    "build_actor_critic",
    "decision_boundary",
    "expert_counterfactual",
    "find_transition_points",
    "gate_attribution",
    "make_market_env",
    "transition_lag",
]
