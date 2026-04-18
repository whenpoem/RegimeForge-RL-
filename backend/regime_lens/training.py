from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
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
    TrainingConfig,
)
from .dqn import DQNAgent
from .hmm_dqn import HMMDQNAgent
from .market import SyntheticMarketEnv
from .metrics import episode_metrics
from .oracle_dqn import OracleDQNAgent
from .rcmoe import RCMoEAgent
from .runtime import configure_runtime


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _create_agent(config: TrainingConfig, resolved_device: str) -> DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent:
    """Instantiate the correct agent type based on config."""
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
    )

    if config.agent_type == AgentType.RCMOE_DQN:
        return RCMoEAgent(
            observation_dim=config.observation_dim,
            n_experts=config.n_experts,
            gate_hidden_dim=config.gate_hidden_dim,
            load_balance_weight=config.load_balance_weight,
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
            **common,
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

    def _config_payload(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["artifact_root"] = str(self.config.artifact_root)
        payload["agent_type"] = self.config.agent_type.value
        payload["curriculum_mode"] = self.config.curriculum_mode.value
        return payload

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

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def _train_loop(self, run_id: str) -> None:
        config = self.config
        runtime = configure_runtime(config.device, config.cpu_threads, config.process_priority)
        self.store.update_run_summary(
            run_id,
            {
                "runtime": runtime.to_payload(),
                "agentType": config.agent_type.value,
            },
        )
        env = SyntheticMarketEnv(config)
        agent = _create_agent(config, runtime.resolved_device)

        metrics_series: list[dict[str, Any]] = []
        checkpoint_summaries: list[dict[str, Any]] = []
        global_step = 0
        started_at = time.perf_counter()

        for episode in range(1, config.episodes + 1):
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

            # HMM: fit detector on warm-up returns each episode
            if _is_hmm(agent):
                agent.fit_detector(env.returns[:config.warmup_steps])

            while not done:
                self._resume_event.wait()
                # Build observation based on agent type
                if _is_oracle(agent):
                    oracle_view = env.observe_oracle()
                    obs_state = oracle_view.state
                elif _is_hmm(agent):
                    base_view = env.observe()
                    recent_ret = env.returns[max(0, env.t - 10) : env.t]
                    obs_state = agent.augment_state(base_view.state, recent_ret)
                else:
                    obs_state = state

                action = agent.select_action(obs_state, epsilon)
                next_state, reward, done, info = env.step(action)
                actions[ACTION_LABELS[action]] += 1

                # Build next observation for storage
                if _is_oracle(agent):
                    next_oracle = env.observe_oracle()
                    next_obs = next_oracle.state
                elif _is_hmm(agent):
                    next_base = env.observe()
                    next_recent = env.returns[max(0, env.t - 10) : env.t]
                    next_obs = agent.augment_state(next_base.state, next_recent)
                else:
                    next_obs = next_state

                agent.store(obs_state, action, reward, next_obs, done)
                global_step += 1
                if global_step >= config.train_after_steps and global_step % config.update_every_steps == 0:
                    for _ in range(config.gradient_steps):
                        loss = agent.update()
                        if loss is not None:
                            loss_values.append(loss)
                episode_reward += reward
                pnl_net = float(info["pnl"]) - float(info["transaction_cost"])
                gross_return += pnl_net
                step_pnls.append(pnl_net)
                step_regimes.append(str(info["regime"]))

                # Update live telemetry for TUI
                with self._lock:
                    self._live_regime = str(info["regime"])
                    self._live_regime_index = int(info["regime_index"])
                    if _is_rcmoe(agent) and agent.last_gate_weights is not None:
                        gw = agent.last_gate_weights
                        self._live_gate_weights = gw.copy()
                        dominant = int(np.argmax(gw))
                        self._live_gate_accuracy_window.append(dominant == self._live_regime_index)
                        if len(self._live_gate_accuracy_window) > 500:
                            self._live_gate_accuracy_window = self._live_gate_accuracy_window[-500:]
                        self._live_expert_history.append({
                            "weights": gw.tolist(),
                            "regime": self._live_regime,
                            "regime_index": self._live_regime_index,
                        })
                        if len(self._live_expert_history) > 500:
                            self._live_expert_history = self._live_expert_history[-500:]

                state = next_state

            # Compute episode-level financial metrics
            fin_metrics = episode_metrics(
                np.asarray(step_pnls), step_regimes, REGIME_LABELS,
            )
            with self._lock:
                self._live_financial_metrics = {
                    k: v for k, v in fin_metrics.items() if isinstance(v, (int, float))
                }

            metric_entry: dict[str, Any] = {
                "episode": episode,
                "globalStep": global_step,
                "epsilon": epsilon,
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
                checkpoint_payload = self._evaluate_checkpoint(agent, env, episode)
                checkpoint_summaries.append(checkpoint_payload["summary"])
                self.store.write_checkpoint(run_id, checkpoint_payload["summary"]["checkpointId"], checkpoint_payload)
                self.store.write_checkpoint_index(run_id, checkpoint_summaries)
                self.store.update_run_summary(
                    run_id,
                    {
                        "checkpoints": [item["checkpointId"] for item in checkpoint_summaries],
                        "latestCheckpointId": checkpoint_payload["summary"]["checkpointId"],
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

    # -----------------------------------------------------------------------
    # Evaluation + checkpoint construction
    # -----------------------------------------------------------------------

    def _evaluate_checkpoint(self, agent: DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent,
                              env: SyntheticMarketEnv, episode: int) -> dict[str, Any]:
        eval_rewards: list[float] = []
        eval_returns: list[float] = []
        random_returns: list[float] = []
        long_returns: list[float] = []
        representative_episode: dict[str, Any] | None = None

        for eval_index, seed in enumerate(self.config.fixed_eval_seeds[: self.config.evaluation_episodes]):
            trace = self._rollout(
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
                agent,
                policy=lambda _state, _step, rng=np.random.default_rng(seed + 31): int(rng.integers(0, len(ACTION_VALUES))),
                seed=seed,
                include_trace=False,
            )
            buy_hold_trace = self._rollout(
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
        }

        # Financial metrics for checkpoint
        if representative_episode.get("financialMetrics"):
            summary["financialMetrics"] = representative_episode["financialMetrics"]

        # Regime analysis for RCMoE agent
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
            expert_analysis = {
                "activation_matrix": regime_analysis["activation_matrix"],
                "expert_utilization": regime_analysis["expert_utilization"],
                "specialisation_score": regime_analysis.get("specialisation_score", 0.0),
                "gate_entropy_per_regime": regime_analysis.get("gate_entropy_per_regime", {}),
            }

        representative_episode["checkpointId"] = checkpoint_id
        representative_episode["episode"] = episode
        # Remove large arrays before serialisation
        representative_episode.pop("gateWeights", None)
        representative_episode.pop("stepRegimes", None)

        result: dict[str, Any] = {
            "summary": summary,
            "episode": representative_episode,
            "policy": self._build_policy_surface(agent, checkpoint_id, episode),
            "embedding": embedding,
        }
        if regime_analysis is not None:
            result["regime_analysis"] = regime_analysis
        if expert_analysis is not None:
            result["expert_analysis"] = expert_analysis
        return result

    def _rollout(
        self,
        agent: DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent,
        policy: Callable[[np.ndarray, int], int],
        seed: int,
        include_trace: bool,
    ) -> dict[str, Any]:
        env = SyntheticMarketEnv(self.config)
        env.reset(seed=seed)

        # HMM: fit on warm-up
        if _is_hmm(agent):
            agent.fit_detector(env.returns[:self.config.warmup_steps])

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

        while not done:
            view = env.observe()

            # Build agent-specific observation
            if _is_oracle(agent):
                oracle_view = env.observe_oracle()
                agent_state = oracle_view.state
            elif _is_hmm(agent):
                recent_ret = env.returns[max(0, env.t - 10) : env.t]
                agent_state = agent.augment_state(view.state, recent_ret)
            else:
                agent_state = view.state

            q_values = agent.q_values(agent_state)
            action = policy(agent_state, step)
            _, reward, done, info = env.step(action)
            action_label = ACTION_LABELS[action]
            regime = str(info["regime"])
            action_counter[action_label] += 1
            regime_counter[regime] += 1
            cumulative_reward += reward
            pnl_net = float(info["pnl"]) - float(info["transaction_cost"])
            strategy_return += pnl_net
            step_pnls.append(pnl_net)
            step_regimes.append(regime)
            states_for_embedding.append(view.state.copy())

            # Collect gate weights for RCMoE
            if _is_rcmoe(agent):
                gw = agent.gate_weights(agent_state)
                gate_weights_list.append(gw.tolist())

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

        fin = episode_metrics(np.asarray(step_pnls), step_regimes, REGIME_LABELS)

        summary = {
            "cumulativeReward": float(cumulative_reward),
            "strategyReturn": float(strategy_return),
            "marketReturn": float((env.prices[env.end_index] / env.prices[self.config.warmup_steps]) - 1.0),
            "actionCounts": {label: int(action_counter[label]) for label in ACTION_LABELS},
            "regimeExposure": {label: int(regime_counter[label]) for label in REGIME_LABELS},
        }

        payload: dict[str, Any] = {
            "summary": summary,
            "financialMetrics": fin,
        }
        if include_trace:
            payload["trace"] = trace_steps
            payload["embeddingStates"] = [state.tolist() for state in states_for_embedding]
        if gate_weights_list:
            payload["gateWeights"] = gate_weights_list
            payload["stepRegimes"] = step_regimes
        return payload

    def _build_policy_surface(self, agent: DQNAgent | RCMoEAgent | OracleDQNAgent | HMMDQNAgent,
                               checkpoint_id: str, episode: int) -> dict[str, Any]:
        env = SyntheticMarketEnv(self.config)
        trend_axis = np.linspace(-2.6, 2.6, self.config.policy_grid_size)
        vol_axis = np.linspace(0.2, 3.2, self.config.policy_grid_size)
        states = np.asarray(
            [
                env.baseline_state(trend_gap_pct=float(trend), volatility_pct=float(volatility))
                for volatility in vol_axis
                for trend in trend_axis
            ],
            dtype=np.float32,
        )

        # For oracle/HMM agents we need to augment states
        if _is_oracle(agent):
            # Use neutral regime (uniform) for policy surface
            n_reg = len(REGIME_LABELS)
            uniform = np.full(n_reg, 1.0 / n_reg, dtype=np.float32)
            states = np.asarray([
                np.concatenate([s, uniform]) for s in states
            ], dtype=np.float32)
        elif _is_hmm(agent):
            n_comp = agent._n_components
            uniform = np.full(n_comp, 1.0 / n_comp, dtype=np.float32)
            states = np.asarray([
                np.concatenate([s, uniform]) for s in states
            ], dtype=np.float32)

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
                # Add gate weights for RCMoE policy surface
                if _is_rcmoe(agent):
                    base_state = states[q_index][:self.config.observation_dim]
                    gw = agent.gate_weights(base_state)
                    cell["gateWeights"] = gw.tolist()
                cells.append(cell)
                q_index += 1
        return {
            "checkpointId": checkpoint_id,
            "episode": episode,
            "axes": {"x": "trendGapPct", "y": "volatilityPct"},
            "cells": cells,
        }

    def _build_embedding(self, episode_payload: dict[str, Any], checkpoint_id: str, episode: int) -> dict[str, Any]:
        states = np.asarray(episode_payload["embeddingStates"], dtype=np.float64)
        if len(states) < 2:
            projected = np.zeros((len(states), 2), dtype=np.float64)
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
