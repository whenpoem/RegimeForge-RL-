from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
import platform
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable

from filelock import FileLock, Timeout
import numpy as np
from sklearn.decomposition import PCA

from .analysis import full_regime_analysis
from .artifacts import ArtifactStore
from .config import (
    ACTION_LABELS,
    ACTION_VALUES,
    FEATURE_NAMES,
    REGIME_LABELS,
    AgentType,
    AlgorithmType,
    GateType,
    TrainingConfig,
    config_from_snapshot,
    config_to_snapshot,
)
from .continuous_agent import ContinuousActorCriticAgent
from .continuous_market import make_market_env
from .context import append_context_state, build_temporal_context, initialise_context_history
from .data import inject_fitted_regime_data
from .dqn import DQNAgent
from .explainability import decision_boundary, expert_counterfactual, gate_attribution, transition_lag
from .hmm_dqn import HMMDQNAgent
from .market import SyntheticMarketEnv
from .metrics import episode_metrics
from .oracle_dqn import OracleDQNAgent
from .rcmoe import RCMoEAgent
from .runtime import configure_runtime
from .tracking import create_tracker
from .transformer_agent import TransformerDQNAgent
from .world_model import WorldModelAgent


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _create_agent(
    config: TrainingConfig,
    resolved_device: str,
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent | ContinuousActorCriticAgent | TransformerDQNAgent | WorldModelAgent:
    """Instantiate the correct agent type based on config."""
    if config.agent_type == AgentType.WORLD_MODEL:
        if observation_dim is None or action_dim is None:
            raise ValueError("World model agent requires explicit observation_dim and action_dim.")
        return WorldModelAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            config=config,
            device=resolved_device,
        )
    if _is_continuous_config(config):
        if observation_dim is None or action_dim is None:
            raise ValueError("Continuous agents require explicit observation_dim and action_dim.")
        if config.agent_type == AgentType.HMM_DQN:
            from .hmm_dqn import HMMContinuousAgent
            return HMMContinuousAgent(
                base_observation_dim=observation_dim,
                action_dim=action_dim,
                config=config,
                device=resolved_device,
                use_hmm=config.use_hmm,
            )
        effective_obs_dim = observation_dim
        if config.agent_type == AgentType.ORACLE_DQN:
            effective_obs_dim = observation_dim + len(REGIME_LABELS)
        return ContinuousActorCriticAgent(
            config,
            observation_dim=effective_obs_dim,
            action_dim=action_dim,
            device=resolved_device,
        )

    common = dict(
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        replay_capacity=config.replay_capacity,
        batch_size=config.batch_size,
        device=resolved_device,
        seed=config.seed,
        use_per=config.use_per,
        per_alpha=config.per_alpha,
        per_epsilon=config.per_epsilon,
        dueling=config.dueling,
        noisy=config.noisy,
    )

    if config.agent_type == AgentType.RCMOE_DQN:
        observation_dim = config.observation_dim
        if config.gate_type == GateType.TEMPORAL:
            observation_dim *= max(config.context_len, 1)
        return RCMoEAgent(
            observation_dim=observation_dim,
            n_experts=config.n_experts,
            gate_hidden_dim=config.gate_hidden_dim,
            load_balance_weight=config.load_balance_weight,
            gate_type=config.gate_type,
            context_len=max(config.context_len, 1),
            hierarchical_moe=config.hierarchical_moe,
            macro_experts=config.macro_experts,
            **common,
        )
    elif config.agent_type == AgentType.ORACLE_DQN:
        return OracleDQNAgent(
            base_observation_dim=config.observation_dim,
            **common,
        )
    elif config.agent_type == AgentType.HMM_DQN:
        return HMMDQNAgent(
            base_observation_dim=config.observation_dim,
            use_hmm=config.use_hmm,
            **common,
        )
    elif config.agent_type == AgentType.TRANSFORMER_DQN:
        return TransformerDQNAgent(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            tau=config.tau,
            replay_capacity=config.replay_capacity,
            batch_size=config.batch_size,
            device=resolved_device,
            seed=config.seed,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            seq_len=config.seq_len,
            dropout=config.dropout,
            use_per=config.use_per,
            per_alpha=config.per_alpha,
            per_epsilon=config.per_epsilon,
        )
    else:
        return DQNAgent(
            observation_dim=config.observation_dim,
            **common,
        )


def _is_rcmoe(agent: object) -> bool:
    return isinstance(agent, RCMoEAgent)


def _is_oracle(agent: object) -> bool:
    return isinstance(agent, OracleDQNAgent)


def _is_hmm(agent: object) -> bool:
    return isinstance(agent, HMMDQNAgent)


def _is_transformer(agent: object) -> bool:
    return isinstance(agent, TransformerDQNAgent)


def _is_world_model(agent: object) -> bool:
    return isinstance(agent, WorldModelAgent)


def _is_hmm_continuous(agent: object) -> bool:
    from .hmm_dqn import HMMContinuousAgent
    return isinstance(agent, HMMContinuousAgent)


def _is_continuous_agent(agent: object) -> bool:
    return isinstance(agent, ContinuousActorCriticAgent)


def _is_continuous_config(config: TrainingConfig) -> bool:
    return bool(config.continuous_actions or config.algorithm != AlgorithmType.DQN)


def _reset_agent_context(agent: object) -> None:
    reset = getattr(agent, "reset_context", None)
    if callable(reset):
        reset()


def _continuous_agent_observation(config: TrainingConfig, agent: object, env: Any, state: np.ndarray) -> np.ndarray:
    if config.agent_type == AgentType.ORACLE_DQN:
        return env.observe_oracle().state
    if _is_hmm_continuous(agent):
        recent_returns = env.returns[max(0, env.t - 10) : env.t]
        return agent.augment_state(state, recent_returns)
    return state


def _continuous_surface_agent_observation(config: TrainingConfig, agent: object, state: np.ndarray) -> np.ndarray:
    if config.agent_type == AgentType.ORACLE_DQN:
        uniform = np.full(len(REGIME_LABELS), 1.0 / len(REGIME_LABELS), dtype=np.float32)
        return np.concatenate([state, uniform]).astype(np.float32)
    if _is_hmm_continuous(agent):
        n_components = int(getattr(agent, "_n_components", len(REGIME_LABELS)))
        uniform = np.full(n_components, 1.0 / n_components, dtype=np.float32)
        return np.concatenate([state, uniform]).astype(np.float32)
    return state


def _uses_temporal_gate(config: TrainingConfig, agent: object) -> bool:
    return _is_rcmoe(agent) and config.gate_type == GateType.TEMPORAL


def _build_env(config: TrainingConfig) -> Any:
    if not _is_continuous_config(config):
        return SyntheticMarketEnv(config)
    asset_names = tuple(config.real_data_symbols) if config.real_data_symbols else ("asset",)
    multi_asset = len(asset_names) > 1
    return make_market_env(config, continuous=True, multi_asset=multi_asset, asset_names=asset_names)


def _market_return(env: Any, config: TrainingConfig) -> float:
    prices = np.asarray(env.prices, dtype=np.float64)
    if prices.ndim == 1:
        return float((prices[env.end_index] / prices[config.warmup_steps]) - 1.0)
    return float(np.mean((prices[env.end_index] / prices[config.warmup_steps]) - 1.0))


def _continuous_action_labels(env: Any) -> list[str]:
    asset_names = list(getattr(env, "asset_names", ()))
    if len(asset_names) == 1:
        return ["allocation"]
    if asset_names:
        return [str(name) for name in asset_names]
    return ["allocation"]


def _continuous_surface_state(env: Any, trend_gap_pct: float, volatility_pct: float) -> np.ndarray:
    if hasattr(env, "asset_names"):
        asset_names = tuple(env.asset_names)
        if len(asset_names) > 1:
            base_env = SyntheticMarketEnv(env.config)
            base_state = base_env.baseline_state(trend_gap_pct=trend_gap_pct, volatility_pct=volatility_pct)
            return np.tile(base_state, len(asset_names)).astype(np.float32)
    baseline_env = SyntheticMarketEnv(env.config if hasattr(env, "config") else TrainingConfig())
    return baseline_env.baseline_state(trend_gap_pct=trend_gap_pct, volatility_pct=volatility_pct)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _representative_trace_state(episode_payload: dict[str, Any]) -> tuple[int, np.ndarray] | None:
    trace = episode_payload.get("trace")
    if not isinstance(trace, list) or not trace:
        return None
    anchor_step = len(trace) // 2
    anchor_state = trace[anchor_step].get("stateVector")
    if not isinstance(anchor_state, list) or not anchor_state:
        return None
    return anchor_step, np.asarray(anchor_state, dtype=np.float32)


def _build_explainability_state(config: TrainingConfig, agent: object, base_state: np.ndarray) -> np.ndarray:
    if _is_oracle(agent) or config.agent_type == AgentType.ORACLE_DQN:
        neutral_regime = np.full(len(REGIME_LABELS), 1.0 / len(REGIME_LABELS), dtype=np.float32)
        return np.concatenate([base_state, neutral_regime]).astype(np.float32)
    if _is_hmm(agent) or _is_hmm_continuous(agent):
        n_components = int(getattr(agent, "_n_components", len(REGIME_LABELS)))
        neutral_posterior = np.full(n_components, 1.0 / n_components, dtype=np.float32)
        return np.concatenate([base_state, neutral_posterior]).astype(np.float32)
    if _is_transformer(agent):
        seq_len = max(int(getattr(agent, "seq_len", config.seq_len)), 1)
        return np.tile(base_state, seq_len).astype(np.float32)
    if _is_rcmoe(agent) and getattr(agent, "gate_type", config.gate_type) == GateType.TEMPORAL:
        context_len = max(int(getattr(agent, "context_len", config.context_len)), 1)
        return np.tile(base_state, context_len).astype(np.float32)
    return np.asarray(base_state, dtype=np.float32)


def _dominant_expert_regime_payload(
    regime_analysis: dict[str, Any] | None,
    gate_weights: np.ndarray,
) -> tuple[list[str], dict[str, str]] | None:
    if regime_analysis is None:
        return None
    activation_matrix = np.asarray(regime_analysis.get("activation_matrix", []), dtype=np.float64)
    if activation_matrix.ndim != 2 or activation_matrix.shape[1] != gate_weights.shape[1]:
        return None
    if activation_matrix.shape[0] != len(REGIME_LABELS):
        return None
    regime_indices = activation_matrix.argmax(axis=0)
    mapping = {
        f"expert_{expert_index}": REGIME_LABELS[int(regime_indices[expert_index])]
        for expert_index in range(gate_weights.shape[1])
    }
    dominant_experts = np.argmax(gate_weights, axis=1)
    inferred_regimes = [mapping[f"expert_{int(expert_index)}"] for expert_index in dominant_experts]
    return inferred_regimes, mapping


# ---------------------------------------------------------------------------
# Training Manager
# ---------------------------------------------------------------------------

class TrainingManager:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.store = ArtifactStore(self.config.artifact_root)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._current_run_id: str | None = None
        self._process_lock: FileLock | None = None
        self._resume_event = threading.Event()
        self._resume_event.set()

        # Live telemetry for TUI consumption (updated every step during training)
        self._live_gate_weights: np.ndarray | None = None
        self._live_regime: str | None = None
        self._live_regime_index: int = -1
        self._live_expert_history: list[dict[str, Any]] = []
        self._live_financial_metrics: dict[str, float] = {}
        self._live_gate_accuracy_window: list[bool] = []

    # -- Live telemetry accessors (thread-safe reads) -----------------------

    @property
    def live_gate_weights(self) -> np.ndarray | None:
        with self._lock:
            if self._live_gate_weights is None:
                return None
            return self._live_gate_weights.copy()

    @property
    def live_regime(self) -> str | None:
        with self._lock:
            return self._live_regime

    @property
    def live_regime_index(self) -> int:
        with self._lock:
            return self._live_regime_index

    @property
    def live_financial_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._live_financial_metrics)

    @property
    def live_gate_accuracy(self) -> float:
        with self._lock:
            window = list(self._live_gate_accuracy_window)
        if not window:
            return 0.0
        return sum(window) / len(window)

    @property
    def live_expert_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._live_expert_history[-200:]]

    @property
    def is_paused(self) -> bool:
        return self.is_training() and not self._resume_event.is_set()

    def live_snapshot(self) -> dict[str, Any]:
        with self._lock:
            gate_weights = None if self._live_gate_weights is None else self._live_gate_weights.copy()
            gate_window = list(self._live_gate_accuracy_window)
            return {
                "gate_weights": gate_weights,
                "regime": self._live_regime,
                "regime_index": self._live_regime_index,
                "financial_metrics": dict(self._live_financial_metrics),
                "gate_accuracy": (sum(gate_window) / len(gate_window)) if gate_window else 0.0,
                "expert_history": [dict(item) for item in self._live_expert_history[-200:]],
                "is_paused": self.is_paused,
            }

    @property
    def current_run_id(self) -> str | None:
        with self._lock:
            return self._current_run_id or self.store.latest_run_id()

    def maybe_autostart(self) -> None:
        if not self.config.autostart:
            return
        with self._lock:
            if self.is_training():
                return
            latest_run_id = self.store.latest_run_id()
            if latest_run_id is None:
                self._current_run_id = None
            elif self._current_run_id is None:
                self._current_run_id = latest_run_id
                return
        if self.current_run_id is None:
            self.start_new_run()

    def is_training(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_new_run(self) -> str:
        with self._lock:
            if self.is_training() and self._current_run_id is not None:
                return self._current_run_id
            if not self._try_acquire_process_lock():
                latest_run_id = self.store.latest_run_id()
                if latest_run_id is not None:
                    self._current_run_id = latest_run_id
                    return latest_run_id
                raise RuntimeError("Another Regime Lens training process is already active.")
            self._resume_event.set()
            self._reset_live_telemetry()
            run_id, _ = self.store.create_run(self._config_payload())
            self._current_run_id = run_id
            self._thread = threading.Thread(target=self._run_train_thread, args=(run_id,), daemon=True, name=f"trainer-{run_id}")
            self._thread.start()
            return run_id

    def run_new_run_blocking(self) -> str:
        with self._lock:
            if self.is_training() and self._current_run_id is not None:
                raise RuntimeError("A training run is already in progress.")
            if not self._try_acquire_process_lock():
                latest_run_id = self.store.latest_run_id()
                if latest_run_id is not None:
                    self._current_run_id = latest_run_id
                    return latest_run_id
                raise RuntimeError("Another Regime Lens training process is already active.")
            self._resume_event.set()
            self._reset_live_telemetry()
            run_id, _ = self.store.create_run(self._config_payload())
            self._current_run_id = run_id
        self._run_train_thread(run_id)
        return run_id

    def pause_training(self) -> bool:
        if not self.is_training():
            return False
        self._resume_event.clear()
        self._update_pause_state(True)
        return True

    def resume_training(self) -> bool:
        if not self.is_training():
            return False
        self._resume_event.set()
        self._update_pause_state(False)
        return True

    def toggle_pause(self) -> bool:
        if self.is_paused:
            self.resume_training()
        else:
            self.pause_training()
        return self.is_paused

    def latest_run_summary(self) -> dict[str, Any] | None:
        run_id = self.current_run_id
        if run_id is None:
            return None
        return self.store.read_run_summary(run_id)

    def metrics(self, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        metrics_path = self.config.artifact_root / resolved / "metrics.json"
        if not metrics_path.exists():
            return None
        payload = self.store.read_json(metrics_path)
        payload["status"] = self.store.read_run_summary(resolved)["status"]
        return payload

    def checkpoints(self, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        index_path = self.config.artifact_root / resolved / "checkpoints" / "index.json"
        if not index_path.exists():
            return None
        return self.store.read_json(index_path)

    def checkpoint_episode(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "episode.json"
        if not path.exists():
            return None
        return self.store.checkpoint_episode(resolved, checkpoint_id)

    def checkpoint_policy(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "policy.json"
        if not path.exists():
            return None
        return self.store.checkpoint_policy(resolved, checkpoint_id)

    def checkpoint_embedding(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "embedding.json"
        if not path.exists():
            return None
        return self.store.checkpoint_embedding(resolved, checkpoint_id)

    def checkpoint_regime_analysis(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "regime_analysis.json"
        if not path.exists():
            return None
        return self.store.checkpoint_regime_analysis(resolved, checkpoint_id)

    def checkpoint_expert_analysis(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "expert_analysis.json"
        if not path.exists():
            return None
        return self.store.checkpoint_expert_analysis(resolved, checkpoint_id)

    def checkpoint_resume_state(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "resume_state.json"
        if not path.exists():
            return None
        return self.store.checkpoint_resume_state(resolved, checkpoint_id)

    def checkpoint_stats(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "stats.json"
        if not path.exists():
            return None
        return self.store.checkpoint_stats(resolved, checkpoint_id)

    def checkpoint_explainability(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "explainability.json"
        if not path.exists():
            return None
        return self.store.checkpoint_explainability(resolved, checkpoint_id)

    def checkpoint_data_fit(self, checkpoint_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        resolved = run_id or self.current_run_id
        if resolved is None:
            return None
        path = self.config.artifact_root / resolved / "checkpoints" / checkpoint_id / "data_fit.json"
        if not path.exists():
            return None
        return self.store.checkpoint_data_fit(resolved, checkpoint_id)

    def load_agent_from_checkpoint(
        self,
        checkpoint_id: str,
        run_id: str | None = None,
        weights_only: bool = True,
        agent_config: TrainingConfig | None = None,
    ) -> DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent | ContinuousActorCriticAgent:
        """Load a trained agent from a saved checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier (e.g. 'ckpt-1200').
            run_id: Run directory name.  Defaults to the current/latest run.
            weights_only: If True, only load network weights (for eval).
                          If False, also restore optimizer state (for resume).

        Returns:
            The agent with loaded weights, ready for evaluation.

        Raises:
            FileNotFoundError: If the checkpoint weights directory does not exist.
        """
        from .runtime import configure_runtime

        resolved = run_id or self.current_run_id
        if resolved is None:
            raise ValueError("No run_id specified and no current run available.")
        if not self.store.has_model_weights(resolved, checkpoint_id):
            raise FileNotFoundError(
                f"No model weights found for checkpoint {checkpoint_id} in run {resolved}"
            )
        saved_config = self._checkpoint_saved_config(resolved, checkpoint_id)
        effective_config = agent_config or replace(
            saved_config,
            device=self.config.device,
            cpu_threads=self.config.cpu_threads,
            process_priority=self.config.process_priority,
        )
        runtime = configure_runtime(
            effective_config.device,
            effective_config.cpu_threads,
            effective_config.process_priority,
        )
        _, agent = self._build_env_and_agent(effective_config, runtime.resolved_device)
        weights_dir = self.store.model_weights_dir(resolved, checkpoint_id)
        agent.load_checkpoint(weights_dir, weights_only=weights_only)
        return agent

    def _build_env_and_agent(
        self,
        config: TrainingConfig,
        resolved_device: str,
    ) -> tuple[Any, DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent | ContinuousActorCriticAgent]:
        env = _build_env(config)
        agent_kwargs: dict[str, int] = {}
        if config.agent_type == AgentType.WORLD_MODEL:
            if _is_continuous_config(config):
                raise ValueError("World model agent currently supports the discrete market environment only.")
            agent_kwargs = {
                "observation_dim": int(config.observation_dim),
                "action_dim": 1,
            }
        elif _is_continuous_config(config):
            agent_kwargs = {
                "observation_dim": int(env.observation_dim),
                "action_dim": int(env.action_dim),
            }
        agent = _create_agent(config, resolved_device, **agent_kwargs)
        return env, agent

    def _resolve_checkpoint_id(self, artifact_root: Any, run_id: str, checkpoint_id: str | None) -> str:
        if checkpoint_id is not None:
            return checkpoint_id
        source_index = self.store.read_json(Path(artifact_root) / run_id / "checkpoints" / "index.json")
        checkpoints = source_index.get("checkpoints", [])
        if not checkpoints:
            raise ValueError(f"Run {run_id} does not contain any checkpoints to resume from.")
        return str(checkpoints[-1]["checkpointId"])

    def _checkpoint_saved_config(self, run_id: str, checkpoint_id: str) -> TrainingConfig:
        resume_state = self.store.checkpoint_resume_state(run_id, checkpoint_id) or {}
        snapshot = resume_state.get("config")
        if not isinstance(snapshot, dict):
            run_summary = self.store.read_run_summary(run_id)
            snapshot = run_summary.get("config")
        if not isinstance(snapshot, dict):
            raise ValueError(f"Run {run_id} is missing its saved configuration snapshot.")
        return config_from_snapshot(snapshot)

    def _merge_resume_config(
        self,
        saved_config: TrainingConfig,
        current_config: TrainingConfig,
        *,
        resolved_checkpoint_id: str,
    ) -> TrainingConfig:
        default_snapshot = config_to_snapshot(TrainingConfig())
        saved_snapshot = config_to_snapshot(saved_config)
        current_snapshot = config_to_snapshot(current_config)
        always_override_fields = {
            "artifact_root",
            "autostart",
            "checkpoint_version",
            "config_path",
            "cpu_threads",
            "device",
            "process_priority",
            "resume_checkpoint_id",
            "resume_run_id",
        }
        safe_explicit_override_fields = {
            "checkpoint_interval",
            "episodes",
            "evaluation_episodes",
            "experiment_name",
            "fixed_eval_seeds",
            "metrics_flush_interval",
            "parallel_workers",
            "policy_grid_size",
            "tracking_backend",
        }

        incompatible_overrides: list[str] = []
        for field_name, current_value in current_snapshot.items():
            if field_name in always_override_fields or field_name in safe_explicit_override_fields:
                continue
            if field_name not in default_snapshot:
                continue
            if current_value == default_snapshot[field_name] or current_value == saved_snapshot.get(field_name):
                continue
            incompatible_overrides.append(field_name)

        if incompatible_overrides:
            incompatible_fields = ", ".join(sorted(incompatible_overrides))
            raise ValueError(
                "Resume configuration overrides incompatible checkpoint settings: "
                f"{incompatible_fields}. "
                "Create a fresh run or resume without changing training/environment-shaping fields."
            )

        merged_snapshot = dict(saved_snapshot)
        for field_name in always_override_fields:
            merged_snapshot[field_name] = current_snapshot[field_name]
        for field_name in safe_explicit_override_fields:
            if current_snapshot[field_name] != default_snapshot[field_name]:
                merged_snapshot[field_name] = current_snapshot[field_name]
        merged_snapshot["resume_checkpoint_id"] = resolved_checkpoint_id
        return config_from_snapshot(merged_snapshot)

    def _build_resume_context(
        self,
        config: TrainingConfig,
    ) -> tuple[TrainingConfig, dict[str, Any] | None, dict[str, Any], str, str]:
        if config.resume_run_id is None:
            raise ValueError("Resume context requested without a resume_run_id.")

        source_run = config.resume_run_id
        source_checkpoint = self._resolve_checkpoint_id(
            config.artifact_root,
            source_run,
            config.resume_checkpoint_id,
        )
        resume_state = self.store.checkpoint_resume_state(source_run, source_checkpoint) or {}
        saved_config = self._checkpoint_saved_config(source_run, source_checkpoint)
        merged_config = self._merge_resume_config(
            saved_config,
            config,
            resolved_checkpoint_id=source_checkpoint,
        )
        data_fit = self.store.checkpoint_data_fit(source_run, source_checkpoint)
        if data_fit is None:
            run_data_fit = Path(config.artifact_root) / source_run / "data_fit.json"
            if run_data_fit.exists():
                data_fit = self.store.read_json(run_data_fit)
        return merged_config, data_fit, resume_state, source_run, source_checkpoint

    def _config_payload(self) -> dict[str, Any]:
        return config_to_snapshot(self.config)

    def _reset_live_telemetry(self) -> None:
        self._live_gate_weights = None
        self._live_regime = None
        self._live_regime_index = -1
        self._live_expert_history = []
        self._live_financial_metrics = {}
        self._live_gate_accuracy_window = []

    def _update_pause_state(self, paused: bool) -> None:
        run_id = self.current_run_id
        if run_id is None:
            return
        self.store.update_run_summary(run_id, {"paused": paused})

    def _run_train_thread(self, run_id: str) -> None:
        try:
            self._train_loop(run_id)
        except Exception as exc:
            self.store.update_run_summary(
                run_id,
                {
                    "status": "failed",
                    "failedAt": datetime.now(tz=UTC).isoformat(),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
        finally:
            self._resume_event.set()
            with self._lock:
                self._thread = None
            self._release_process_lock()

    def _try_acquire_process_lock(self) -> bool:
        lock_path = self.config.artifact_root / "active-training.lock"
        process_lock = FileLock(str(lock_path))
        try:
            process_lock.acquire(timeout=0)
        except Timeout:
            return False
        self._process_lock = process_lock
        return True

    def _release_process_lock(self) -> None:
        if self._process_lock is None:
            return
        try:
            self._process_lock.release()
        except Exception:
            pass
        finally:
            self._process_lock = None

    def _build_embedding(self, episode_payload: dict[str, Any], checkpoint_id: str, episode: int) -> dict[str, Any]:
        states = np.asarray(episode_payload["embeddingStates"], dtype=np.float64)
        if states.ndim == 1:
            states = states.reshape(-1, 1)
        if len(states) < 2:
            projected = np.zeros((len(states), 2), dtype=np.float64)
        elif states.shape[1] < 2:
            projected = np.column_stack([states[:, 0], np.zeros(len(states), dtype=np.float64)])
        else:
            projected = PCA(n_components=2).fit_transform(states)

        points: list[dict[str, Any]] = []
        for idx, projected_state in enumerate(projected):
            trace_point = episode_payload["trace"][idx]
            point: dict[str, Any] = {
                "step": idx,
                "x": float(projected_state[0]),
                "y": float(projected_state[1]),
                "regime": trace_point["regime"],
                "action": trace_point["action"],
                "position": trace_point["position"],
                "reward": trace_point["reward"],
                "price": trace_point["price"],
            }
            if "gateWeights" in trace_point:
                point["gateWeights"] = trace_point["gateWeights"]
            points.append(point)
        return {
            "checkpointId": checkpoint_id,
            "episode": episode,
            "points": points,
        }

    @staticmethod
    def _build_repro_bundle(config: TrainingConfig, runtime: Any, tracker: Any | None = None) -> dict[str, Any]:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "runtime": runtime.to_payload(),
            "checkpointVersion": config.checkpoint_version,
            "trackingBackend": config.tracking_backend.value,
            "tracking": tracker.status_payload() if tracker is not None and hasattr(tracker, "status_payload") else None,
        }

    def _train_loop(self, run_id: str) -> None:
        resume_state: dict[str, Any] = {}
        source_run: str | None = None
        source_checkpoint: str | None = None
        if self.config.resume_run_id is not None:
            config, data_fit, resume_state, source_run, source_checkpoint = self._build_resume_context(self.config)
        else:
            config, data_fit = inject_fitted_regime_data(self.config)
        runtime = configure_runtime(config.device, config.cpu_threads, config.process_priority)
        run_dir = config.artifact_root / run_id
        tracker = create_tracker(config, run_dir)
        metrics_series: list[dict[str, Any]] = []
        checkpoint_summaries: list[dict[str, Any]] = []
        global_step = 0
        start_episode = 1
        elapsed_offset = 0.0

        self.store.update_run_summary(
            run_id,
            {
                "runtime": runtime.to_payload(),
                "agentType": config.agent_type.value,
                "config": config_to_snapshot(config),
                "tracking": tracker.status_payload(),
            },
        )
        if data_fit is not None:
            self.store.write_run_file(run_id, "data_fit.json", data_fit)

        env, agent = self._build_env_and_agent(config, runtime.resolved_device)
        if source_run is not None and source_checkpoint is not None:
            agent.load_checkpoint(
                self.store.model_weights_dir(source_run, source_checkpoint),
                weights_only=False,
            )
            start_episode = int(resume_state.get("episode", 0)) + 1
            global_step = int(resume_state.get("globalStep", 0))
            elapsed_offset = float(resume_state.get("elapsedSeconds", 0.0))
            if config.episodes < start_episode:
                raise ValueError(
                    f"Resume target episodes ({config.episodes}) must exceed the checkpoint episode "
                    f"({start_episode - 1})."
                )
            self.store.update_run_summary(
                run_id,
                {
                    "resumedFromRunId": source_run,
                    "resumedFromCheckpointId": source_checkpoint,
                },
            )

        if _is_world_model(agent):
            self._train_loop_world_model(
                run_id,
                config,
                runtime,
                agent,
                env,
                tracker,
                data_fit=data_fit,
                global_step=global_step,
                start_episode=start_episode,
                elapsed_offset=elapsed_offset,
            )
            return

        if _is_continuous_config(config):
            self._train_loop_continuous(
                run_id,
                config,
                runtime,
                agent,
                env,
                tracker,
                data_fit=data_fit,
                global_step=global_step,
                start_episode=start_episode,
                elapsed_offset=elapsed_offset,
            )
            return

        started_at = time.perf_counter() - elapsed_offset
        try:
            for episode in range(start_episode, config.episodes + 1):
                epsilon = config.epsilon_for_episode(episode)
                state = env.reset(seed=config.seed + episode)
                done = False
                episode_reward = 0.0
                market_return = (env.prices[env.end_index] / env.prices[config.warmup_steps]) - 1.0
                loss_values: list[float] = []
                actions = Counter()
                gross_return = 0.0
                step_pnls: list[float] = []
                step_regimes: list[str] = []
                context_len = max(int(getattr(agent, "context_len", config.context_len)), 1)
                uses_temporal_gate = _is_rcmoe(agent) and getattr(agent, "gate_type", config.gate_type) == GateType.TEMPORAL
                context_history = initialise_context_history(state, context_len=context_len)

                if _is_hmm(agent):
                    agent.fit_detector(env.returns[:config.warmup_steps])
                if _is_transformer(agent):
                    agent.reset_context()

                while not done:
                    self._resume_event.wait()
                    if _is_oracle(agent):
                        obs_state = env.observe_oracle().state
                    elif _is_hmm(agent):
                        recent_ret = env.returns[max(0, env.t - 10) : env.t]
                        obs_state = agent.augment_state(env.observe().state, recent_ret)
                    elif _is_transformer(agent):
                        obs_state = state  # transformer handles context internally
                    elif uses_temporal_gate:
                        obs_state = build_temporal_context(context_history, context_len=context_len)
                    else:
                        obs_state = state

                    action = agent.select_action(obs_state, epsilon)
                    next_state, reward, done, info = env.step(action)
                    actions[ACTION_LABELS[action]] += 1
                    next_context_history = append_context_state(context_history, next_state)

                    if _is_oracle(agent):
                        next_obs = env.observe_oracle().state
                    elif _is_hmm(agent):
                        next_recent = env.returns[max(0, env.t - 10) : env.t]
                        next_obs = agent.augment_state(env.observe().state, next_recent)
                    elif uses_temporal_gate:
                        next_obs = build_temporal_context(next_context_history, context_len=context_len)
                    else:
                        next_obs = next_state

                    agent.store(obs_state, action, reward, next_obs, done)
                    global_step += 1
                    if global_step >= config.train_after_steps and global_step % config.update_every_steps == 0:
                        per_beta = min(
                            1.0,
                            config.per_beta_start + (1.0 - config.per_beta_start) * (episode / max(config.episodes, 1)),
                        )
                        for _ in range(config.gradient_steps):
                            loss = agent.update(per_beta=per_beta)
                            if loss is not None:
                                loss_values.append(loss)

                    episode_reward += reward
                    pnl_net = float(info["pnl"]) - float(info["transaction_cost"])
                    gross_return += pnl_net
                    step_pnls.append(pnl_net)
                    step_regimes.append(str(info["regime"]))

                    with self._lock:
                        self._live_regime = str(info["regime"])
                        self._live_regime_index = int(info["regime_index"])
                        if _is_rcmoe(agent) and agent.last_gate_weights is not None:
                            gate_weights = agent.last_gate_weights
                            self._live_gate_weights = gate_weights.copy()
                            dominant = int(np.argmax(gate_weights))
                            self._live_gate_accuracy_window.append(dominant == self._live_regime_index)
                            if len(self._live_gate_accuracy_window) > 500:
                                self._live_gate_accuracy_window = self._live_gate_accuracy_window[-500:]
                            self._live_expert_history.append(
                                {
                                    "weights": gate_weights.tolist(),
                                    "regime": self._live_regime,
                                    "regime_index": self._live_regime_index,
                                }
                            )
                            if len(self._live_expert_history) > 500:
                                self._live_expert_history = self._live_expert_history[-500:]

                    state = next_state
                    context_history = next_context_history

                fin_metrics = episode_metrics(np.asarray(step_pnls), step_regimes, REGIME_LABELS)
                with self._lock:
                    self._live_financial_metrics = {
                        key: value for key, value in fin_metrics.items() if isinstance(value, (int, float))
                    }

                metric_entry: dict[str, Any] = {
                    "episode": episode,
                    "globalStep": global_step,
                    "epsilon": epsilon,
                    "perBeta": min(
                        1.0,
                        config.per_beta_start + (1.0 - config.per_beta_start) * (episode / max(config.episodes, 1)),
                    ),
                    "totalReward": episode_reward,
                    "strategyReturn": gross_return,
                    "marketReturn": market_return,
                    "avgLoss": float(np.mean(loss_values)) if loss_values else None,
                    "actionMix": {label: int(actions[label]) for label in ACTION_LABELS},
                    "sharpe": fin_metrics.get("sharpe"),
                    "sortino": fin_metrics.get("sortino"),
                    "maxDrawdown": fin_metrics.get("max_drawdown"),
                    "winRate": fin_metrics.get("win_rate"),
                }
                if _is_rcmoe(agent):
                    metric_entry["gateAccuracy"] = self.live_gate_accuracy
                metrics_series.append(metric_entry)
                tracker.log_episode(episode, metric_entry)

                self.store.update_run_summary(
                    run_id,
                    {
                        "currentEpisode": episode,
                        "latestCheckpointId": checkpoint_summaries[-1]["checkpointId"] if checkpoint_summaries else None,
                        "episodeLength": config.episode_length,
                        "featureNames": FEATURE_NAMES,
                        "actionLabels": ACTION_LABELS,
                        "regimeLabels": REGIME_LABELS,
                        "episodesPlanned": config.episodes,
                        "globalStep": global_step,
                        "elapsedSeconds": time.perf_counter() - started_at,
                        "agentType": config.agent_type.value,
                    },
                )

                if episode % config.metrics_flush_interval == 0 or episode == config.episodes:
                    self.store.write_metrics(run_id, metrics_series)

                if episode % config.checkpoint_interval == 0 or episode == config.episodes:
                    checkpoint_payload = self._evaluate_checkpoint(
                        config,
                        agent,
                        episode,
                        global_step=global_step,
                        elapsed_seconds=time.perf_counter() - started_at,
                        data_fit=data_fit,
                        repro=self._build_repro_bundle(config, runtime, tracker),
                    )
                    checkpoint_summaries.append(checkpoint_payload["summary"])
                    checkpoint_id = checkpoint_payload["summary"]["checkpointId"]
                    self.store.write_checkpoint(run_id, checkpoint_id, checkpoint_payload)
                    agent.save_checkpoint(self.store.model_weights_dir(run_id, checkpoint_id))
                    tracker.log_checkpoint(episode, checkpoint_payload["summary"])
                    self.store.write_checkpoint_index(run_id, checkpoint_summaries)
                    self.store.update_run_summary(
                        run_id,
                        {
                            "checkpoints": [item["checkpointId"] for item in checkpoint_summaries],
                            "latestCheckpointId": checkpoint_id,
                        },
                    )

            self.store.write_metrics(run_id, metrics_series)
            self.store.update_run_summary(
                run_id,
                {
                    "status": "completed",
                    "completedAt": datetime.now(tz=UTC).isoformat(),
                    "currentEpisode": config.episodes,
                    "globalStep": global_step,
                    "elapsedSeconds": time.perf_counter() - started_at,
                },
            )
        finally:
            tracker.close()

    def _train_loop_continuous(
        self,
        run_id: str,
        config: TrainingConfig,
        runtime: Any,
        agent: ContinuousActorCriticAgent,
        env: Any,
        tracker: Any,
        *,
        data_fit: dict[str, Any] | None,
        global_step: int,
        start_episode: int,
        elapsed_offset: float,
    ) -> None:
        started_at = time.perf_counter() - elapsed_offset
        metrics_series: list[dict[str, Any]] = []
        checkpoint_summaries: list[dict[str, Any]] = []
        action_labels = _continuous_action_labels(env)

        try:
            for episode in range(start_episode, config.episodes + 1):
                state = env.reset(seed=config.seed + episode)
                if hasattr(agent, 'reset_context'):
                    agent.reset_context()
                done = False
                episode_reward = 0.0
                loss_values: list[float] = []
                step_pnls: list[float] = []
                step_regimes: list[str] = []
                net_exposures: list[float] = []
                gross_exposures: list[float] = []
                last_agent_observation: np.ndarray | None = None

                is_oracle_continuous = config.agent_type == AgentType.ORACLE_DQN
                is_hmm_continuous = _is_hmm_continuous(agent)

                if is_hmm_continuous and hasattr(agent, 'fit_detector'):
                    agent.fit_detector(env.returns[:config.warmup_steps])

                while not done:
                    self._resume_event.wait()
                    if is_oracle_continuous:
                        obs_for_agent = env.observe_oracle().state
                    elif is_hmm_continuous:
                        recent_ret = env.returns[max(0, env.t - 10) : env.t]
                        obs_for_agent = agent.augment_state(state, recent_ret)
                    else:
                        obs_for_agent = state
                    last_agent_observation = obs_for_agent
                    action_info = agent.act(obs_for_agent, deterministic=False)
                    action = np.asarray(action_info["action"], dtype=np.float32).reshape(-1)
                    next_state, reward, done, info = env.step(action)
                    if is_oracle_continuous:
                        next_obs_for_agent = env.observe_oracle().state
                    elif is_hmm_continuous:
                        next_recent_ret = env.returns[max(0, env.t - 10) : env.t]
                        next_obs_for_agent = agent.augment_state(next_state, next_recent_ret)
                    else:
                        next_obs_for_agent = next_state
                    agent.store(
                        obs_for_agent,
                        action,
                        reward,
                        next_obs_for_agent,
                        done,
                        log_prob=action_info.get("log_prob"),
                        value=action_info.get("value"),
                    )
                    global_step += 1

                    update_metrics = None
                    if config.algorithm == AlgorithmType.SAC:
                        update_metrics = agent.update()
                    if update_metrics is not None:
                        numeric_losses = [
                            float(value)
                            for key, value in update_metrics.items()
                            if isinstance(value, (int, float)) and key.endswith("loss")
                        ]
                        if numeric_losses:
                            loss_values.append(float(np.mean(numeric_losses)))

                    episode_reward += float(reward)
                    pnl_net = float(info["pnl"]) - float(info["transaction_cost"])
                    step_pnls.append(pnl_net)
                    step_regimes.append(str(info["regime"]))
                    positions = np.asarray(info.get("positions", action), dtype=np.float32).reshape(-1)
                    net_exposures.append(float(positions.sum()))
                    gross_exposures.append(float(np.abs(positions).sum()))

                    with self._lock:
                        self._live_regime = str(info["regime"])
                        self._live_regime_index = int(info["regime_index"])
                        gate_weights = agent.gate_weights(obs_for_agent)
                        if gate_weights is not None:
                            gate_weights = gate_weights.astype(np.float64, copy=False)
                            self._live_gate_weights = gate_weights
                            dominant = int(np.argmax(gate_weights))
                            self._live_gate_accuracy_window.append(dominant == self._live_regime_index)
                            if len(self._live_gate_accuracy_window) > 500:
                                self._live_gate_accuracy_window = self._live_gate_accuracy_window[-500:]
                            self._live_expert_history.append(
                                {
                                    "weights": gate_weights.tolist(),
                                    "regime": self._live_regime,
                                    "regime_index": self._live_regime_index,
                                }
                            )
                            if len(self._live_expert_history) > 500:
                                self._live_expert_history = self._live_expert_history[-500:]

                    state = next_state

                if config.algorithm == AlgorithmType.PPO:
                    update_metrics = agent.update()
                    if update_metrics is not None:
                        numeric_losses = [
                            float(value)
                            for key, value in update_metrics.items()
                            if isinstance(value, (int, float)) and key.endswith("loss")
                        ]
                        if numeric_losses:
                            loss_values.append(float(np.mean(numeric_losses)))

                fin_metrics = episode_metrics(np.asarray(step_pnls, dtype=np.float64), step_regimes, REGIME_LABELS)
                with self._lock:
                    self._live_financial_metrics = {
                        key: value for key, value in fin_metrics.items() if isinstance(value, (int, float))
                    }

                metric_entry: dict[str, Any] = {
                    "episode": episode,
                    "globalStep": global_step,
                    "epsilon": None,
                    "totalReward": float(episode_reward),
                    "strategyReturn": float(np.sum(step_pnls)),
                    "marketReturn": _market_return(env, config),
                    "avgLoss": float(np.mean(loss_values)) if loss_values else None,
                    "actionMix": {
                        "meanNetExposure": float(np.mean(net_exposures)) if net_exposures else 0.0,
                        "meanGrossExposure": float(np.mean(gross_exposures)) if gross_exposures else 0.0,
                    },
                    "sharpe": fin_metrics.get("sharpe"),
                    "sortino": fin_metrics.get("sortino"),
                    "maxDrawdown": fin_metrics.get("max_drawdown"),
                    "winRate": fin_metrics.get("win_rate"),
                }
                if last_agent_observation is not None and agent.gate_weights(last_agent_observation) is not None:
                    metric_entry["gateAccuracy"] = self.live_gate_accuracy
                metrics_series.append(metric_entry)
                tracker.log_episode(episode, metric_entry)

                self.store.update_run_summary(
                    run_id,
                    {
                        "currentEpisode": episode,
                        "latestCheckpointId": checkpoint_summaries[-1]["checkpointId"] if checkpoint_summaries else None,
                        "episodeLength": config.episode_length,
                        "featureNames": list(FEATURE_NAMES),
                        "actionLabels": action_labels,
                        "regimeLabels": REGIME_LABELS,
                        "episodesPlanned": config.episodes,
                        "globalStep": global_step,
                        "elapsedSeconds": time.perf_counter() - started_at,
                        "agentType": config.agent_type.value,
                    },
                )

                if episode % config.metrics_flush_interval == 0 or episode == config.episodes:
                    self.store.write_metrics(run_id, metrics_series)

                if episode % config.checkpoint_interval == 0 or episode == config.episodes:
                    checkpoint_payload = self._evaluate_continuous_checkpoint(
                        config,
                        agent,
                        episode,
                        global_step=global_step,
                        elapsed_seconds=time.perf_counter() - started_at,
                        data_fit=data_fit,
                        repro=self._build_repro_bundle(config, runtime, tracker),
                    )
                    checkpoint_summaries.append(checkpoint_payload["summary"])
                    checkpoint_id = checkpoint_payload["summary"]["checkpointId"]
                    self.store.write_checkpoint(run_id, checkpoint_id, checkpoint_payload)
                    agent.save_checkpoint(self.store.model_weights_dir(run_id, checkpoint_id))
                    tracker.log_checkpoint(episode, checkpoint_payload["summary"])
                    self.store.write_checkpoint_index(run_id, checkpoint_summaries)
                    self.store.update_run_summary(
                        run_id,
                        {
                            "checkpoints": [item["checkpointId"] for item in checkpoint_summaries],
                            "latestCheckpointId": checkpoint_id,
                        },
                    )

            self.store.write_metrics(run_id, metrics_series)
            self.store.update_run_summary(
                run_id,
                {
                    "status": "completed",
                    "completedAt": datetime.now(tz=UTC).isoformat(),
                    "currentEpisode": config.episodes,
                    "globalStep": global_step,
                    "elapsedSeconds": time.perf_counter() - started_at,
                },
            )
        finally:
            tracker.close()

    def _train_loop_world_model(
        self,
        run_id: str,
        config: TrainingConfig,
        runtime: Any,
        agent: WorldModelAgent,
        env: Any,
        tracker: Any,
        *,
        data_fit: dict[str, Any] | None,
        global_step: int,
        start_episode: int,
        elapsed_offset: float,
    ) -> None:
        started_at = time.perf_counter() - elapsed_offset
        metrics_series: list[dict[str, Any]] = []
        checkpoint_summaries: list[dict[str, Any]] = []

        self.store.update_run_summary(
            run_id,
            {
                "runtime": runtime.to_payload(),
                "agentType": config.agent_type.value,
                "config": config_to_snapshot(config),
                "tracking": tracker.status_payload(),
            },
        )

        try:
            for episode in range(start_episode, config.episodes + 1):
                state = env.reset(seed=config.seed + episode)
                agent.reset()
                done = False
                episode_reward = 0.0
                step_pnls: list[float] = []
                step_regimes: list[str] = []

                while not done:
                    self._resume_event.wait()
                    action_info = agent.act(state, deterministic=False)
                    action = np.clip(
                        np.asarray(action_info["action"], dtype=np.float32).reshape(-1),
                        -1.0, 1.0,
                    )
                    # Map continuous action to discrete for the discrete env
                    action_idx = 1  # flat
                    if action[0] > 0.33:
                        action_idx = 2  # long
                    elif action[0] < -0.33:
                        action_idx = 0  # short

                    next_state, reward, done, info = env.step(action_idx)
                    agent.store_transition(state, action, reward, done)
                    global_step += 1
                    episode_reward += float(reward)
                    pnl_net = float(info["pnl"]) - float(info["transaction_cost"])
                    step_pnls.append(pnl_net)
                    step_regimes.append(str(info["regime"]))
                    state = next_state

                # Train world model + actor/critic after episode
                update_metrics = agent.update()
                if update_metrics is None:
                    update_metrics = {}

                fin_metrics = episode_metrics(np.asarray(step_pnls, dtype=np.float64), step_regimes, REGIME_LABELS)
                metric_entry: dict[str, Any] = {
                    "episode": episode,
                    "globalStep": global_step,
                    "epsilon": None,
                    "totalReward": float(episode_reward),
                    "strategyReturn": float(np.sum(step_pnls)),
                    "marketReturn": _market_return(env, config),
                    "sharpe": fin_metrics.get("sharpe"),
                    "sortino": fin_metrics.get("sortino"),
                    "maxDrawdown": fin_metrics.get("max_drawdown"),
                    "winRate": fin_metrics.get("win_rate"),
                }
                if update_metrics:
                    metric_entry["worldModelLoss"] = update_metrics.get("world_loss")
                    metric_entry["avgLoss"] = update_metrics.get("actor_loss")
                metrics_series.append(metric_entry)
                tracker.log_episode(episode, metric_entry)

                self.store.update_run_summary(
                    run_id,
                    {
                        "currentEpisode": episode,
                        "latestCheckpointId": checkpoint_summaries[-1]["checkpointId"] if checkpoint_summaries else None,
                        "episodeLength": config.episode_length,
                        "featureNames": list(FEATURE_NAMES),
                        "actionLabels": ["short", "flat", "long"],
                        "regimeLabels": REGIME_LABELS,
                        "episodesPlanned": config.episodes,
                        "globalStep": global_step,
                        "elapsedSeconds": time.perf_counter() - started_at,
                        "agentType": config.agent_type.value,
                    },
                )

                if episode % config.metrics_flush_interval == 0 or episode == config.episodes:
                    self.store.write_metrics(run_id, metrics_series)

                if episode % config.checkpoint_interval == 0 or episode == config.episodes:
                    checkpoint_id = f"ckpt-{episode:04d}"
                    summary: dict[str, Any] = {
                        "checkpointId": checkpoint_id,
                        "episode": episode,
                        "capturedAt": datetime.now(tz=UTC).isoformat(),
                        "avgEvalReward": float(episode_reward),
                        "agentReturn": float(np.sum(step_pnls)),
                        "financialMetrics": fin_metrics,
                    }
                    checkpoint_payload = {
                        "summary": summary,
                        "episode": {"steps": [], "financialMetrics": fin_metrics},
                        "policy": {"cells": []},
                        "embedding": {"states": []},
                    }
                    checkpoint_summaries.append(summary)
                    self.store.write_checkpoint(run_id, checkpoint_id, checkpoint_payload)
                    agent.save_checkpoint(self.store.model_weights_dir(run_id, checkpoint_id))
                    tracker.log_checkpoint(episode, summary)
                    self.store.write_checkpoint_index(run_id, checkpoint_summaries)
                    self.store.update_run_summary(
                        run_id,
                        {
                            "checkpoints": [item["checkpointId"] for item in checkpoint_summaries],
                            "latestCheckpointId": checkpoint_id,
                        },
                    )

            self.store.write_metrics(run_id, metrics_series)
            self.store.update_run_summary(
                run_id,
                {
                    "status": "completed",
                    "completedAt": datetime.now(tz=UTC).isoformat(),
                    "currentEpisode": config.episodes,
                    "globalStep": global_step,
                    "elapsedSeconds": time.perf_counter() - started_at,
                },
            )
        finally:
            tracker.close()

    def _evaluate_continuous_checkpoint(
        self,
        config: TrainingConfig,
        agent: ContinuousActorCriticAgent,
        episode: int,
        *,
        global_step: int,
        elapsed_seconds: float,
        data_fit: dict[str, Any] | None,
        repro: dict[str, Any],
    ) -> dict[str, Any]:
        eval_rewards: list[float] = []
        eval_returns: list[float] = []
        random_returns: list[float] = []
        long_returns: list[float] = []
        representative_episode: dict[str, Any] | None = None
        evaluation_seeds = config.fixed_eval_seeds[: config.evaluation_episodes]

        for eval_index, seed in enumerate(evaluation_seeds):
            random_rng = np.random.default_rng(seed + 31)

            def random_continuous_policy(
                _state: np.ndarray,
                _step: int,
                *,
                rng: np.random.Generator = random_rng,
                action_dim: int = agent.action_dim,
            ) -> np.ndarray:
                return rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)

            trace = self._rollout_continuous(
                config,
                agent,
                policy=lambda state, _step: np.asarray(agent.act(state, deterministic=True)["action"], dtype=np.float32).reshape(-1),
                seed=seed,
                include_trace=eval_index == 0,
            )
            eval_rewards.append(trace["summary"]["cumulativeReward"])
            eval_returns.append(trace["summary"]["strategyReturn"])
            if eval_index == 0:
                representative_episode = trace

            random_trace = self._rollout_continuous(
                config,
                agent,
                policy=random_continuous_policy,
                seed=seed,
                include_trace=False,
            )
            long_trace = self._rollout_continuous(
                config,
                agent,
                policy=lambda _state, _step, action_dim=agent.action_dim: np.ones(action_dim, dtype=np.float32),
                seed=seed,
                include_trace=False,
            )
            random_returns.append(random_trace["summary"]["strategyReturn"])
            long_returns.append(long_trace["summary"]["strategyReturn"])

        if representative_episode is None:
            raise RuntimeError("Failed to build representative continuous evaluation episode.")

        checkpoint_id = f"ckpt-{episode:04d}"
        embedding = self._build_embedding(representative_episode, checkpoint_id, episode)
        representative_episode.pop("embeddingStates", None)

        summary: dict[str, Any] = {
            "checkpointId": checkpoint_id,
            "episode": episode,
            "capturedAt": datetime.now(tz=UTC).isoformat(),
            "avgEvalReward": float(np.mean(eval_rewards)),
            "agentReturn": float(np.mean(eval_returns)),
            "randomReturn": float(np.mean(random_returns)),
            "buyHoldReturn": float(np.mean(long_returns)),
            "actionMix": representative_episode["summary"]["actionCounts"],
        }
        if representative_episode.get("financialMetrics"):
            summary["financialMetrics"] = representative_episode["financialMetrics"]

        regime_analysis: dict[str, Any] | None = None
        expert_analysis: dict[str, Any] | None = None
        if representative_episode.get("gateWeights"):
            gw = np.asarray(representative_episode["gateWeights"], dtype=np.float64)
            regimes = representative_episode.get("stepRegimes", [])
            hidden = None
            if representative_episode.get("embeddingStates"):
                hidden = np.asarray(representative_episode["embeddingStates"], dtype=np.float64)
            regime_analysis = full_regime_analysis(gw, regimes, REGIME_LABELS, hidden)
            summary["nmi"] = regime_analysis["nmi"]
            summary["ari"] = regime_analysis["ari"]
            summary["gateEntropy"] = regime_analysis["gate_entropy"]
            summary["specialisationScore"] = regime_analysis.get("specialisation_score", 0.0)
            summary["expertUtilization"] = regime_analysis["expert_utilization"]
            summary["gateType"] = config.gate_type.value
            summary["contextLen"] = int(config.context_len)
            expert_analysis = {
                "activation_matrix": regime_analysis["activation_matrix"],
                "expert_utilization": regime_analysis["expert_utilization"],
                "specialisation_score": regime_analysis.get("specialisation_score", 0.0),
                "gate_entropy_per_regime": regime_analysis.get("gate_entropy_per_regime", {}),
            }

        explainability = self._build_explainability_bundle(
            config,
            agent,
            representative_episode,
            regime_analysis=regime_analysis,
        )
        representative_episode["checkpointId"] = checkpoint_id
        representative_episode["episode"] = episode
        representative_episode.pop("gateWeights", None)
        representative_episode.pop("stepRegimes", None)

        result: dict[str, Any] = {
            "summary": summary,
            "episode": representative_episode,
            "policy": self._build_continuous_policy_surface(config, agent, checkpoint_id, episode),
            "embedding": embedding,
            "resume_state": {
                "episode": episode,
                "globalStep": global_step,
                "elapsedSeconds": elapsed_seconds,
                "checkpointVersion": config.checkpoint_version,
                "config": config_to_snapshot(config),
            },
            "stats": {
                "evaluationSeeds": list(evaluation_seeds),
                "evalRewardMean": float(np.mean(eval_rewards)),
                "evalRewardStd": float(np.std(eval_rewards, ddof=0)),
                "agentReturnMean": float(np.mean(eval_returns)),
                "agentReturnStd": float(np.std(eval_returns, ddof=0)),
                "randomReturnMean": float(np.mean(random_returns)),
                "buyHoldReturnMean": float(np.mean(long_returns)),
            },
            "repro": repro,
        }
        if data_fit is not None:
            result["data_fit"] = data_fit
        if regime_analysis is not None:
            result["regime_analysis"] = regime_analysis
        if expert_analysis is not None:
            result["expert_analysis"] = expert_analysis
        if explainability is not None:
            result["explainability"] = explainability
        return result

    def _rollout_continuous(
        self,
        config: TrainingConfig,
        agent: ContinuousActorCriticAgent,
        policy: Callable[[np.ndarray, int], np.ndarray],
        seed: int,
        include_trace: bool,
    ) -> dict[str, Any]:
        env = _build_env(config)
        state = env.reset(seed=seed)
        if _is_hmm_continuous(agent):
            agent.fit_detector(env.returns[:config.warmup_steps])
        _reset_agent_context(agent)
        done = False
        step = 0
        cumulative_reward = 0.0
        strategy_return = 0.0
        trace_steps: list[dict[str, Any]] = []
        states_for_embedding: list[np.ndarray] = []
        gate_weights_list: list[list[float]] = []
        step_regimes: list[str] = []
        step_pnls: list[float] = []
        net_exposures: list[float] = []
        gross_exposures: list[float] = []

        while not done:
            view = env.observe()
            obs_for_policy = _continuous_agent_observation(config, agent, env, state)
            action = np.asarray(policy(obs_for_policy, step), dtype=np.float32).reshape(-1)
            next_state, reward, done, info = env.step(action)
            positions = np.asarray(info.get("positions", action), dtype=np.float32).reshape(-1)
            regime = str(info["regime"])
            pnl_net = float(info["pnl"]) - float(info["transaction_cost"])

            cumulative_reward += float(reward)
            strategy_return += pnl_net
            step_pnls.append(pnl_net)
            step_regimes.append(regime)
            net_exposures.append(float(positions.sum()))
            gross_exposures.append(float(np.abs(positions).sum()))
            states_for_embedding.append(agent.hidden_activations(obs_for_policy))

            gate_weights = agent.gate_weights(obs_for_policy)
            if gate_weights is not None:
                gate_weights_list.append(gate_weights.tolist())

            if include_trace:
                current_prices = info.get("prices", info.get("price", 0.0))
                next_prices = info.get("next_prices", info.get("next_price", 0.0))
                trace_entry: dict[str, Any] = {
                    "step": step,
                    "price": float(np.mean(np.asarray(current_prices, dtype=np.float64))),
                    "nextPrice": float(np.mean(np.asarray(next_prices, dtype=np.float64))),
                    "regime": regime,
                    "action": positions.tolist() if positions.size > 1 else float(positions[0]),
                    "position": positions.tolist() if positions.size > 1 else float(positions[0]),
                    "reward": float(reward),
                    "pnl": float(info["pnl"]),
                    "transactionCost": float(info["transaction_cost"]),
                    "holdPenalty": float(info["hold_penalty"]),
                    "qValues": {},
                    "featureMap": {key: float(value) for key, value in view.feature_map.items()},
                    "stateVector": [float(value) for value in view.state.tolist()],
                }
                if gate_weights is not None:
                    trace_entry["gateWeights"] = gate_weights.tolist()
                trace_steps.append(trace_entry)

            state = next_state
            step += 1

        financial_metrics = episode_metrics(np.asarray(step_pnls, dtype=np.float64), step_regimes, REGIME_LABELS)
        payload: dict[str, Any] = {
            "summary": {
                "cumulativeReward": float(cumulative_reward),
                "strategyReturn": float(strategy_return),
                "marketReturn": _market_return(env, config),
                "actionCounts": {
                    "meanNetExposure": float(np.mean(net_exposures)) if net_exposures else 0.0,
                    "meanGrossExposure": float(np.mean(gross_exposures)) if gross_exposures else 0.0,
                },
                "regimeExposure": {label: int(sum(step_regime == label for step_regime in step_regimes)) for label in REGIME_LABELS},
            },
            "financialMetrics": financial_metrics,
        }
        if include_trace:
            payload["trace"] = trace_steps
            payload["embeddingStates"] = [state.tolist() for state in states_for_embedding]
        if gate_weights_list:
            payload["gateWeights"] = gate_weights_list
            payload["stepRegimes"] = step_regimes
        return payload

    def _build_continuous_policy_surface(
        self,
        config: TrainingConfig,
        agent: ContinuousActorCriticAgent,
        checkpoint_id: str,
        episode: int,
    ) -> dict[str, Any]:
        surface_env = _build_env(config)
        trend_axis = np.linspace(-2.6, 2.6, config.policy_grid_size)
        vol_axis = np.linspace(0.2, 3.2, config.policy_grid_size)
        cells: list[dict[str, Any]] = []

        for volatility in vol_axis:
            for trend in trend_axis:
                state = _continuous_surface_state(surface_env, trend_gap_pct=float(trend), volatility_pct=float(volatility))
                agent_state = _continuous_surface_agent_observation(config, agent, state)
                _reset_agent_context(agent)
                action = np.asarray(agent.act(agent_state, deterministic=True)["action"], dtype=np.float32).reshape(-1)
                cell: dict[str, Any] = {
                    "trendGapPct": float(trend),
                    "volatilityPct": float(volatility),
                    "action": action.tolist() if action.size > 1 else float(action[0]),
                }
                gate_weights = agent.gate_weights(agent_state)
                if gate_weights is not None:
                    cell["gateWeights"] = gate_weights.tolist()
                cells.append(cell)
        _reset_agent_context(agent)

        return {
            "checkpointId": checkpoint_id,
            "episode": episode,
            "axes": {"x": "trendGapPct", "y": "volatilityPct"},
            "cells": cells,
        }

    def _evaluate_checkpoint(
        self,
        config: TrainingConfig,
        agent: DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent,
        episode: int,
        *,
        global_step: int,
        elapsed_seconds: float,
        data_fit: dict[str, Any] | None,
        repro: dict[str, Any],
    ) -> dict[str, Any]:
        eval_rewards: list[float] = []
        eval_returns: list[float] = []
        random_returns: list[float] = []
        long_returns: list[float] = []
        representative_episode: dict[str, Any] | None = None

        evaluation_seeds = config.fixed_eval_seeds[: config.evaluation_episodes]
        for eval_index, seed in enumerate(evaluation_seeds):
            random_rng = np.random.default_rng(seed + 31)

            def random_discrete_policy(
                _state: np.ndarray,
                _step: int,
                *,
                rng: np.random.Generator = random_rng,
            ) -> int:
                return int(rng.integers(0, len(ACTION_VALUES)))

            trace = self._rollout(
                config,
                agent,
                policy=lambda state, _step: agent.greedy_action(state),
                seed=seed,
                include_trace=eval_index == 0,
            )
            eval_rewards.append(trace["summary"]["cumulativeReward"])
            eval_returns.append(trace["summary"]["strategyReturn"])
            if eval_index == 0:
                representative_episode = trace

            random_trace = self._rollout(
                config,
                agent,
                policy=random_discrete_policy,
                seed=seed,
                include_trace=False,
            )
            buy_hold_trace = self._rollout(
                config,
                agent,
                policy=lambda _state, _step: 2,
                seed=seed,
                include_trace=False,
            )
            random_returns.append(random_trace["summary"]["strategyReturn"])
            long_returns.append(buy_hold_trace["summary"]["strategyReturn"])

        if representative_episode is None:
            raise RuntimeError("Failed to build representative evaluation episode.")

        checkpoint_id = f"ckpt-{episode:04d}"
        embedding = self._build_embedding(representative_episode, checkpoint_id, episode)
        representative_episode.pop("embeddingStates", None)

        summary: dict[str, Any] = {
            "checkpointId": checkpoint_id,
            "episode": episode,
            "capturedAt": datetime.now(tz=UTC).isoformat(),
            "avgEvalReward": float(np.mean(eval_rewards)),
            "agentReturn": float(np.mean(eval_returns)),
            "randomReturn": float(np.mean(random_returns)),
            "buyHoldReturn": float(np.mean(long_returns)),
            "actionMix": representative_episode["summary"]["actionCounts"],
            "usePer": bool(config.use_per),
            "perAlpha": float(config.per_alpha),
            "perBetaStart": float(config.per_beta_start),
        }
        if representative_episode.get("financialMetrics"):
            summary["financialMetrics"] = representative_episode["financialMetrics"]

        regime_analysis: dict[str, Any] | None = None
        expert_analysis: dict[str, Any] | None = None
        if _is_rcmoe(agent) and representative_episode.get("gateWeights"):
            gw = np.asarray(representative_episode["gateWeights"], dtype=np.float64)
            regimes = representative_episode.get("stepRegimes", [])
            hidden = None
            if representative_episode.get("embeddingStates"):
                hidden = np.asarray(representative_episode["embeddingStates"], dtype=np.float64)
            regime_analysis = full_regime_analysis(gw, regimes, REGIME_LABELS, hidden)
            summary["nmi"] = regime_analysis["nmi"]
            summary["ari"] = regime_analysis["ari"]
            summary["gateEntropy"] = regime_analysis["gate_entropy"]
            summary["specialisationScore"] = regime_analysis.get("specialisation_score", 0.0)
            summary["expertUtilization"] = regime_analysis["expert_utilization"]
            summary["gateType"] = getattr(agent, "gate_type", config.gate_type).value
            summary["contextLen"] = int(getattr(agent, "context_len", config.context_len))
            expert_analysis = {
                "activation_matrix": regime_analysis["activation_matrix"],
                "expert_utilization": regime_analysis["expert_utilization"],
                "specialisation_score": regime_analysis.get("specialisation_score", 0.0),
                "gate_entropy_per_regime": regime_analysis.get("gate_entropy_per_regime", {}),
            }

        explainability = self._build_explainability_bundle(
            config,
            agent,
            representative_episode,
            regime_analysis=regime_analysis,
        )
        representative_episode["checkpointId"] = checkpoint_id
        representative_episode["episode"] = episode
        representative_episode.pop("gateWeights", None)
        representative_episode.pop("stepRegimes", None)

        result: dict[str, Any] = {
            "summary": summary,
            "episode": representative_episode,
            "policy": self._build_policy_surface(config, agent, checkpoint_id, episode),
            "embedding": embedding,
            "resume_state": {
                "episode": episode,
                "globalStep": global_step,
                "elapsedSeconds": elapsed_seconds,
                "checkpointVersion": config.checkpoint_version,
                "config": config_to_snapshot(config),
            },
            "stats": {
                "evaluationSeeds": list(evaluation_seeds),
                "evalRewardMean": float(np.mean(eval_rewards)),
                "evalRewardStd": float(np.std(eval_rewards, ddof=0)),
                "agentReturnMean": float(np.mean(eval_returns)),
                "agentReturnStd": float(np.std(eval_returns, ddof=0)),
                "randomReturnMean": float(np.mean(random_returns)),
                "buyHoldReturnMean": float(np.mean(long_returns)),
            },
            "repro": repro,
        }
        if data_fit is not None:
            result["data_fit"] = data_fit
        if regime_analysis is not None:
            result["regime_analysis"] = regime_analysis
        if expert_analysis is not None:
            result["expert_analysis"] = expert_analysis
        if explainability is not None:
            result["explainability"] = explainability
        return result

    def _build_explainability_bundle(
        self,
        config: TrainingConfig,
        agent: DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent,
        representative_episode: dict[str, Any],
        *,
        regime_analysis: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        representative = _representative_trace_state(representative_episode)
        if representative is None:
            return None

        anchor_step, anchor_state = representative
        model_state = _build_explainability_state(config, agent, anchor_state)
        grid_size = max(9, min(int(config.policy_grid_size), 21))
        payload: dict[str, Any] = {
            "anchorStep": anchor_step,
            "anchorState": anchor_state.tolist(),
            "anchorModelState": model_state.tolist(),
            "featureNames": list(FEATURE_NAMES),
            "boundaryAxes": {
                "x": FEATURE_NAMES[3],
                "y": FEATURE_NAMES[2],
            },
            "policyBoundary": _jsonable(
                decision_boundary(
                    agent,
                    model_state,
                    feature_x=3,
                    feature_y=2,
                    grid_size=grid_size,
                )
            ),
        }

        if not representative_episode.get("gateWeights"):
            return payload

        gate_weights = np.asarray(representative_episode["gateWeights"], dtype=np.float64)
        if gate_weights.ndim != 2 or gate_weights.size == 0:
            return payload

        anchor_index = min(anchor_step, gate_weights.shape[0] - 1)
        target_expert = int(np.argmax(gate_weights[anchor_index]))
        payload["anchorGateWeights"] = gate_weights[anchor_index].tolist()
        payload["gateAttribution"] = _jsonable(
            gate_attribution(
                agent,
                model_state,
                steps=24,
                expert_index=target_expert,
            )
        )
        payload["gateBoundary"] = _jsonable(
            decision_boundary(
                agent,
                model_state,
                feature_x=3,
                feature_y=2,
                grid_size=grid_size,
                target="gate",
            )
        )
        payload["expertCounterfactuals"] = [
            _jsonable(
                {
                    "expertIndex": expert_index,
                    **expert_counterfactual(agent, model_state, expert_index=expert_index),
                }
            )
            for expert_index in range(int(gate_weights.shape[1]))
        ]

        regime_payload = _dominant_expert_regime_payload(regime_analysis, gate_weights)
        step_regimes = representative_episode.get("stepRegimes", [])
        if regime_payload is not None and isinstance(step_regimes, list) and step_regimes:
            inferred_regimes, expert_map = regime_payload
            payload["dominantExpertRegimeMap"] = expert_map
            payload["transitionLag"] = _jsonable(
                transition_lag(
                    step_regimes,
                    inferred_regimes,
                    max_lag=min(12, max(3, len(step_regimes) // 8)),
                )
            )
        return payload

    def _rollout(
        self,
        config: TrainingConfig,
        agent: DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent,
        policy: Callable[[np.ndarray, int], int],
        seed: int,
        include_trace: bool,
    ) -> dict[str, Any]:
        env = SyntheticMarketEnv(config)
        initial_state = env.reset(seed=seed)
        if _is_hmm(agent):
            agent.fit_detector(env.returns[:config.warmup_steps])
        _reset_agent_context(agent)

        uses_temporal_gate = _is_rcmoe(agent) and getattr(agent, "gate_type", config.gate_type) == GateType.TEMPORAL
        context_len = max(int(getattr(agent, "context_len", config.context_len)), 1)
        done = False
        step = 0
        cumulative_reward = 0.0
        strategy_return = 0.0
        action_counter = Counter()
        regime_counter = Counter()
        trace_steps: list[dict[str, Any]] = []
        states_for_embedding: list[np.ndarray] = []
        gate_weights_list: list[list[float]] = []
        step_regimes: list[str] = []
        step_pnls: list[float] = []
        context_history = initialise_context_history(initial_state, context_len=context_len)

        while not done:
            view = env.observe()
            if _is_oracle(agent):
                agent_state = env.observe_oracle().state
            elif _is_hmm(agent):
                recent_ret = env.returns[max(0, env.t - 10) : env.t]
                agent_state = agent.augment_state(view.state, recent_ret)
            elif uses_temporal_gate:
                agent_state = build_temporal_context(context_history, context_len=context_len)
            else:
                agent_state = view.state

            q_values = agent.q_values(agent_state)
            action = policy(agent_state, step)
            next_state, reward, done, info = env.step(action)
            context_history = append_context_state(context_history, next_state)

            action_label = ACTION_LABELS[action]
            regime = str(info["regime"])
            action_counter[action_label] += 1
            regime_counter[regime] += 1
            cumulative_reward += float(reward)
            pnl_net = float(info["pnl"]) - float(info["transaction_cost"])
            strategy_return += pnl_net
            step_pnls.append(pnl_net)
            step_regimes.append(regime)
            states_for_embedding.append(agent.hidden_activations(agent_state))

            if _is_rcmoe(agent):
                gate_weights = agent.gate_weights(agent_state)
                gate_weights_list.append(gate_weights.tolist())

            if include_trace:
                trace_entry: dict[str, Any] = {
                    "step": step,
                    "price": float(info["price"]),
                    "nextPrice": float(info["next_price"]),
                    "regime": regime,
                    "action": action_label,
                    "position": int(info["position"]),
                    "reward": float(reward),
                    "pnl": float(info["pnl"]),
                    "transactionCost": float(info["transaction_cost"]),
                    "holdPenalty": float(info["hold_penalty"]),
                    "qValues": {
                        ACTION_LABELS[index]: float(value)
                        for index, value in enumerate(q_values.tolist())
                    },
                    "featureMap": {key: float(value) for key, value in view.feature_map.items()},
                    "stateVector": [float(value) for value in view.state.tolist()],
                }
                if _is_rcmoe(agent) and gate_weights_list:
                    trace_entry["gateWeights"] = gate_weights_list[-1]
                trace_steps.append(trace_entry)
            step += 1

        financial_metrics = episode_metrics(np.asarray(step_pnls), step_regimes, REGIME_LABELS)
        payload: dict[str, Any] = {
            "summary": {
                "cumulativeReward": float(cumulative_reward),
                "strategyReturn": float(strategy_return),
                "marketReturn": float((env.prices[env.end_index] / env.prices[config.warmup_steps]) - 1.0),
                "actionCounts": {label: int(action_counter[label]) for label in ACTION_LABELS},
                "regimeExposure": {label: int(regime_counter[label]) for label in REGIME_LABELS},
            },
            "financialMetrics": financial_metrics,
        }
        if include_trace:
            payload["trace"] = trace_steps
            payload["embeddingStates"] = [state.tolist() for state in states_for_embedding]
        if gate_weights_list:
            payload["gateWeights"] = gate_weights_list
            payload["stepRegimes"] = step_regimes
        return payload

    def _build_policy_surface(
        self,
        config: TrainingConfig,
        agent: DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent,
        checkpoint_id: str,
        episode: int,
    ) -> dict[str, Any]:
        env = SyntheticMarketEnv(config)
        trend_axis = np.linspace(-2.6, 2.6, config.policy_grid_size)
        vol_axis = np.linspace(0.2, 3.2, config.policy_grid_size)
        states = np.asarray(
            [
                env.baseline_state(trend_gap_pct=float(trend), volatility_pct=float(volatility))
                for volatility in vol_axis
                for trend in trend_axis
            ],
            dtype=np.float32,
        )

        uses_temporal_gate = _is_rcmoe(agent) and getattr(agent, "gate_type", config.gate_type) == GateType.TEMPORAL
        context_len = max(int(getattr(agent, "context_len", config.context_len)), 1)
        if _is_oracle(agent):
            uniform = np.full(len(REGIME_LABELS), 1.0 / len(REGIME_LABELS), dtype=np.float32)
            states = np.asarray([np.concatenate([state, uniform]) for state in states], dtype=np.float32)
        elif _is_hmm(agent):
            uniform = np.full(agent._n_components, 1.0 / agent._n_components, dtype=np.float32)
            states = np.asarray([np.concatenate([state, uniform]) for state in states], dtype=np.float32)
        elif _is_transformer(agent):
            seq_len = max(int(getattr(agent, "seq_len", config.seq_len)), 1)
            states = np.asarray([np.tile(state, seq_len) for state in states], dtype=np.float32)
        elif uses_temporal_gate:
            states = np.asarray([np.tile(state, context_len) for state in states], dtype=np.float32)

        q_matrix = agent.batch_q_values(states)
        cells: list[dict[str, Any]] = []
        q_index = 0
        for volatility in vol_axis:
            for trend in trend_axis:
                q_values = q_matrix[q_index]
                best_index = int(np.argmax(q_values))
                cell: dict[str, Any] = {
                    "trendGapPct": float(trend),
                    "volatilityPct": float(volatility),
                    "bestAction": ACTION_LABELS[best_index],
                    "qValues": {
                        ACTION_LABELS[index]: float(value)
                        for index, value in enumerate(q_values.tolist())
                    },
                }
                if _is_rcmoe(agent):
                    cell["gateWeights"] = agent.gate_weights(states[q_index]).tolist()
                cells.append(cell)
                q_index += 1
        return {
            "checkpointId": checkpoint_id,
            "episode": episode,
            "axes": {"x": "trendGapPct", "y": "volatilityPct"},
            "cells": cells,
        }
