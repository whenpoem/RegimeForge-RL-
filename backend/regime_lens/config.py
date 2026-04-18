from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


REGIME_LABELS = ("bull", "bear", "chop", "shock")
ACTION_VALUES = (-1, 0, 1)
ACTION_LABELS = ("short", "flat", "long")
FEATURE_NAMES = (
    "short_return_pct",
    "medium_return_pct",
    "volatility_pct",
    "trend_gap_pct",
    "drawdown_pct",
    "position",
    "unrealized_pnl_pct",
)


class AgentType(str, Enum):
    """Supported RL agent architectures."""
    DQN = "dqn"
    RCMOE_DQN = "rcmoe_dqn"
    ORACLE_DQN = "oracle_dqn"
    HMM_DQN = "hmm_dqn"


class CurriculumMode(str, Enum):
    """Regime sampling strategies for training episodes."""
    NATURAL = "natural"         # Markov chain (original)
    UNIFORM = "uniform"         # Force equal regime exposure
    ADVERSARIAL = "adversarial" # Oversample worst-performing regime


@dataclass(slots=True)
class TrainingConfig:
    # ---- Experiment identity ----
    experiment_name: str = "default"
    agent_type: AgentType = AgentType.DQN
    curriculum_mode: CurriculumMode = CurriculumMode.NATURAL

    # ---- Multi-seed support ----
    seeds: tuple[int, ...] = (17, 42, 123, 456, 789)

    # ---- Environment ----
    artifact_root: Path = Path(__file__).resolve().parents[1] / "artifacts"
    seed: int = 17
    episode_length: int = 224
    warmup_steps: int = 28
    episodes: int = 1200
    slippage_bps: float = 0.0  # basis points of slippage per trade

    # ---- Training loop ----
    checkpoint_interval: int = 100
    metrics_flush_interval: int = 20
    evaluation_episodes: int = 4
    replay_capacity: int = 60_000
    batch_size: int = 192
    gamma: float = 0.985
    learning_rate: float = 8e-4
    hidden_dim: int = 128
    train_after_steps: int = 1_500
    update_every_steps: int = 4
    gradient_steps: int = 1
    tau: float = 0.02
    transaction_cost: float = 0.00065
    position_penalty: float = 0.00006

    # ---- RCMoE-specific ----
    n_experts: int = 4
    gate_hidden_dim: int = 64
    load_balance_weight: float = 0.01

    # ---- Runtime ----
    device: str = "auto"
    cpu_threads: int | None = None
    process_priority: str = "below_normal"
    policy_grid_size: int = 21
    autostart: bool = True

    # ---- Evaluation ----
    fixed_eval_seeds: tuple[int, ...] = (9001, 9017, 9031, 9049, 9067, 9091)

    # ---- Regime dynamics ----
    regime_transition: tuple[tuple[float, ...], ...] = (
        (0.88, 0.03, 0.08, 0.01),
        (0.03, 0.87, 0.06, 0.04),
        (0.11, 0.09, 0.73, 0.07),
        (0.18, 0.19, 0.23, 0.40),
    )
    regime_params: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "bull": {"drift": 0.0010, "vol": 0.0080, "autocorr": 0.28, "jump_prob": 0.0, "jump_scale": 0.0},
            "bear": {"drift": -0.0012, "vol": 0.0105, "autocorr": 0.22, "jump_prob": 0.0, "jump_scale": 0.0},
            "chop": {"drift": 0.0001, "vol": 0.0052, "autocorr": -0.32, "jump_prob": 0.0, "jump_scale": 0.0},
            "shock": {"drift": -0.0003, "vol": 0.0200, "autocorr": -0.48, "jump_prob": 0.12, "jump_scale": 0.032},
        }
    )

    @property
    def observation_dim(self) -> int:
        return len(FEATURE_NAMES)

    @property
    def oracle_observation_dim(self) -> int:
        """Observation dim when regime one-hot is appended."""
        return len(FEATURE_NAMES) + len(REGIME_LABELS)

    @property
    def action_dim(self) -> int:
        return len(ACTION_VALUES)

    @property
    def n_regimes(self) -> int:
        return len(REGIME_LABELS)

    def epsilon_for_episode(self, episode: int) -> float:
        progress = min(max((episode - 1) / max(self.episodes - 1, 1), 0.0), 1.0)
        floor = 0.05
        return max(floor, 1.0 - progress * 0.92)
