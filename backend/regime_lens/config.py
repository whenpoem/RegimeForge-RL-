from __future__ import annotations

from dataclasses import fields, is_dataclass, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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


class GateType(str, Enum):
    """Supported RCMoE gate implementations."""

    MLP = "mlp"
    TEMPORAL = "temporal"


class DataSource(str, Enum):
    """How to derive the regime parameters used by the synthetic environment."""

    SYNTHETIC = "synthetic"
    CSV = "csv"


class AlgorithmType(str, Enum):
    """Top-level learning algorithm family."""

    DQN = "dqn"
    PPO = "ppo"
    SAC = "sac"


class TrackingBackend(str, Enum):
    """Supported experiment tracking backends."""

    NONE = "none"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"


@dataclass(slots=True)
class TrainingConfig:
    # ---- Experiment identity ----
    experiment_name: str = "default"
    agent_type: AgentType = AgentType.DQN
    curriculum_mode: CurriculumMode = CurriculumMode.NATURAL
    algorithm: AlgorithmType = AlgorithmType.DQN
    config_path: Path | None = None
    checkpoint_version: int = 2

    # ---- Multi-seed support ----
    seeds: tuple[int, ...] = (17, 42, 123, 456, 789)
    parallel_workers: int = 1

    # ---- Environment ----
    artifact_root: Path = Path(__file__).resolve().parents[1] / "artifacts"
    seed: int = 17
    episode_length: int = 224
    warmup_steps: int = 28
    episodes: int = 1200
    slippage_bps: float = 0.0  # basis points of slippage per trade
    continuous_actions: bool = False
    data_source: DataSource = DataSource.SYNTHETIC
    data_cache_path: Path | None = None
    real_data_symbols: tuple[str, ...] = ("SPY", "QQQ", "GLD")
    nonstationary_mode: str = "stationary"
    nonstationary_drift_scale: float = 0.0

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
    resume_run_id: str | None = None
    resume_checkpoint_id: str | None = None

    # ---- Prioritized Experience Replay ----
    use_per: bool = True
    per_alpha: float = 0.6       # prioritisation exponent (0 = uniform)
    per_beta_start: float = 0.4  # initial importance-sampling correction
    per_epsilon: float = 1e-6    # small constant to prevent zero priority

    # ---- RCMoE-specific ----
    n_experts: int = 4
    gate_hidden_dim: int = 64
    load_balance_weight: float = 0.01
    gate_type: GateType = GateType.MLP
    context_len: int = 8
    hierarchical_moe: bool = False
    macro_experts: int = 2
    dueling: bool = False
    noisy: bool = False

    # ---- Runtime ----
    device: str = "auto"
    cpu_threads: int | None = None
    process_priority: str = "below_normal"
    policy_grid_size: int = 21
    autostart: bool = True
    tracking_backend: TrackingBackend = TrackingBackend.TENSORBOARD

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


def config_to_snapshot(config: TrainingConfig) -> dict[str, Any]:
    return {
        field.name: _snapshot_value(getattr(config, field.name))
        for field in fields(config)
    }


def config_from_snapshot(snapshot: dict[str, Any]) -> TrainingConfig:
    payload = dict(snapshot)
    enum_fields: dict[str, type[Enum]] = {
        "agent_type": AgentType,
        "curriculum_mode": CurriculumMode,
        "gate_type": GateType,
        "data_source": DataSource,
        "algorithm": AlgorithmType,
        "tracking_backend": TrackingBackend,
    }
    tuple_int_fields = {"seeds", "fixed_eval_seeds"}
    tuple_str_fields = {"real_data_symbols"}
    path_fields = {"artifact_root", "data_cache_path", "config_path"}

    for key, enum_type in enum_fields.items():
        if isinstance(payload.get(key), str):
            payload[key] = enum_type(payload[key])

    for key in tuple_int_fields:
        if isinstance(payload.get(key), list):
            payload[key] = tuple(int(value) for value in payload[key])

    for key in tuple_str_fields:
        if isinstance(payload.get(key), list):
            payload[key] = tuple(str(value) for value in payload[key])

    for key in path_fields:
        value = payload.get(key)
        if value is not None and not isinstance(value, Path):
            payload[key] = Path(value)

    if isinstance(payload.get("regime_transition"), list):
        payload["regime_transition"] = tuple(tuple(float(value) for value in row) for row in payload["regime_transition"])

    allowed = {k: v for k, v in payload.items() if k in TrainingConfig.__dataclass_fields__}
    return TrainingConfig(**allowed)


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _snapshot_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    return value
