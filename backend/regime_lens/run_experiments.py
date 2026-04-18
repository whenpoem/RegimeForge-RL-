"""Experiment runner for Regime Lens / RCMoE research runs.

This module provides a practical CLI for:
- planning experiment matrices without executing them,
- running the full benchmark matrix or smaller subsets,
- executing ablation and OOD generalization sweeps,
- aggregating results from training artifacts, and
- exporting a compact report bundle.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from backend.regime_lens.config import ACTION_LABELS, REGIME_LABELS, AgentType, CurriculumMode, TrainingConfig
    from backend.regime_lens.experiment import ExperimentResult, SeedResult, results_to_latex
    from backend.regime_lens.market import SyntheticMarketEnv
    from backend.regime_lens.metrics import episode_metrics
    from backend.regime_lens.training import TrainingManager
else:
    from .config import ACTION_LABELS, REGIME_LABELS, AgentType, CurriculumMode, TrainingConfig
    from .experiment import ExperimentResult, SeedResult, results_to_latex
    from .market import SyntheticMarketEnv
    from .metrics import episode_metrics
    from .training import TrainingManager


DEFAULT_SEEDS: tuple[int, ...] = (17, 42, 123, 456, 789)
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (9001, 9017, 9031, 9049, 9067, 9091)
DEFAULT_ABLATION_EXPERTS: tuple[int, ...] = (2, 3, 4, 6, 8)
DEFAULT_ABLATION_GATE_HIDDEN: tuple[int, ...] = (32, 64, 128, 256)
DEFAULT_ABLATION_LB_WEIGHTS: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1)
DEFAULT_ABLATION_HIDDEN_DIMS: tuple[int, ...] = (64, 128, 256)
UTC = timezone.utc


@dataclass(slots=True)
class ExperimentSpec:
    key: str
    method_name: str
    kind: str
    description: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    baseline_policy: str | None = None
    tags: tuple[str, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "methodName": self.method_name,
            "kind": self.kind,
            "description": self.description,
            "configOverrides": _jsonable(self.config_overrides),
            "baselinePolicy": self.baseline_policy,
            "tags": list(self.tags),
        }


@dataclass(slots=True)
class SuitePlan:
    name: str
    specs: list[ExperimentSpec]
    base_config: TrainingConfig
    description: str = ""

    @property
    def total_runs(self) -> int:
        return len(self.specs) * len(self.base_config.seeds)

    @property
    def eval_seed_count(self) -> int:
        return len(self.base_config.fixed_eval_seeds)


@dataclass(slots=True)
class ExecutionRecord:
    spec_key: str
    method_name: str
    seed: int
    status: str
    run_id: str | None = None
    checkpoint_id: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None


def _build_full_suite(base_config: TrainingConfig) -> SuitePlan:
    specs = [
        ExperimentSpec(
            key="vanilla_dqn",
            method_name="Vanilla DQN",
            kind="trained",
            description="Standard DQN baseline.",
            config_overrides={"agent_type": AgentType.DQN},
            tags=("full", "baseline"),
        ),
        ExperimentSpec(
            key="oracle_dqn",
            method_name="Oracle DQN",
            kind="trained",
            description="Upper-bound baseline with true regime one-hot appended.",
            config_overrides={"agent_type": AgentType.ORACLE_DQN},
            tags=("full", "baseline"),
        ),
        ExperimentSpec(
            key="rcmoe_dqn",
            method_name="RCMoE-DQN",
            kind="trained",
            description="Mixture-of-experts DQN without load-balancing penalty.",
            config_overrides={"agent_type": AgentType.RCMOE_DQN, "load_balance_weight": 0.0},
            tags=("full", "rcmoe"),
        ),
        ExperimentSpec(
            key="rcmoe_dqn_lb",
            method_name="RCMoE-DQN+LB",
            kind="trained",
            description="Mixture-of-experts DQN with load-balancing penalty.",
            config_overrides={"agent_type": AgentType.RCMOE_DQN, "load_balance_weight": 0.01},
            tags=("full", "rcmoe"),
        ),
        ExperimentSpec(
            key="hmm_dqn",
            method_name="HMM+DQN",
            kind="trained",
            description="Two-stage regime detector plus DQN policy.",
            config_overrides={"agent_type": AgentType.HMM_DQN},
            tags=("full", "baseline"),
        ),
        ExperimentSpec(
            key="random",
            method_name="Random",
            kind="baseline",
            description="Stochastic action baseline.",
            baseline_policy="random",
            tags=("full", "baseline"),
        ),
        ExperimentSpec(
            key="buy_hold",
            method_name="Buy&Hold",
            kind="baseline",
            description="Always-long benchmark.",
            baseline_policy="buy_hold",
            tags=("full", "baseline"),
        ),
    ]
    return SuitePlan("full", specs, base_config, "Core method matrix: 7 methods x 5 seeds x 6 eval seeds.")


def _build_smoke_suite(base_config: TrainingConfig) -> SuitePlan:
    smoke_config = replace(
        base_config,
        episodes=min(base_config.episodes, 20),
        evaluation_episodes=min(base_config.evaluation_episodes, 2),
        checkpoint_interval=max(5, min(base_config.checkpoint_interval, 10)),
        metrics_flush_interval=max(5, min(base_config.metrics_flush_interval, 5)),
        seeds=base_config.seeds[:1],
        fixed_eval_seeds=base_config.fixed_eval_seeds[:2],
    )
    specs = [
        ExperimentSpec(
            key="smoke_vanilla",
            method_name="Vanilla DQN",
            kind="trained",
            description="Quick smoke-test run.",
            config_overrides={"agent_type": AgentType.DQN},
            tags=("smoke",),
        ),
        ExperimentSpec(
            key="smoke_rcmoe",
            method_name="RCMoE-DQN+LB",
            kind="trained",
            description="Quick smoke-test run for the MoE pipeline.",
            config_overrides={"agent_type": AgentType.RCMOE_DQN, "load_balance_weight": 0.01},
            tags=("smoke", "rcmoe"),
        ),
        ExperimentSpec(
            key="smoke_random",
            method_name="Random",
            kind="baseline",
            description="Quick stochastic baseline.",
            baseline_policy="random",
            tags=("smoke", "baseline"),
        ),
    ]
    return SuitePlan("smoke", specs, smoke_config, "Short smoke suite for pipeline validation.")


def _build_ablation_suite(base_config: TrainingConfig, kind: str = "all") -> SuitePlan:
    specs: list[ExperimentSpec] = []
    kind = kind.lower()

    if kind in {"all", "experts"}:
        for n_experts in DEFAULT_ABLATION_EXPERTS:
            specs.append(
                ExperimentSpec(
                    key=f"experts_{n_experts}",
                    method_name=f"RCMoE-{n_experts}E",
                    kind="trained",
                    description="Expert count ablation.",
                    config_overrides={
                        "agent_type": AgentType.RCMOE_DQN,
                        "n_experts": n_experts,
                        "load_balance_weight": 0.01,
                    },
                    tags=("ablation", "experts"),
                )
            )

    if kind in {"all", "gate"}:
        for hidden_dim in DEFAULT_ABLATION_GATE_HIDDEN:
            specs.append(
                ExperimentSpec(
                    key=f"gate_{hidden_dim}",
                    method_name=f"Gate-{hidden_dim}",
                    kind="trained",
                    description="Gate capacity ablation via hidden dimension.",
                    config_overrides={
                        "agent_type": AgentType.RCMOE_DQN,
                        "gate_hidden_dim": hidden_dim,
                        "load_balance_weight": 0.01,
                    },
                    tags=("ablation", "gate"),
                )
            )

    if kind in {"all", "lb"}:
        for weight in DEFAULT_ABLATION_LB_WEIGHTS:
            specs.append(
                ExperimentSpec(
                    key=f"lb_{str(weight).replace('.', 'p')}",
                    method_name=f"LB-{weight:g}",
                    kind="trained",
                    description="Load-balancing penalty sweep.",
                    config_overrides={
                        "agent_type": AgentType.RCMOE_DQN,
                        "load_balance_weight": weight,
                    },
                    tags=("ablation", "lb"),
                )
            )

    if kind in {"all", "hidden"}:
        for hidden_dim in DEFAULT_ABLATION_HIDDEN_DIMS:
            specs.append(
                ExperimentSpec(
                    key=f"hidden_{hidden_dim}",
                    method_name=f"Hidden-{hidden_dim}",
                    kind="trained",
                    description="Shared hidden dimension sweep.",
                    config_overrides={
                        "agent_type": AgentType.RCMOE_DQN,
                        "hidden_dim": hidden_dim,
                        "load_balance_weight": 0.01,
                    },
                    tags=("ablation", "hidden"),
                )
            )

    return SuitePlan(f"ablation-{kind}", specs, base_config, "RCMoE ablation sweeps.")


def _build_ood_suite(base_config: TrainingConfig, kind: str = "all") -> SuitePlan:
    specs: list[ExperimentSpec] = []
    kind = kind.lower()

    base_specs = _build_full_suite(base_config).specs

    def _with_env(spec: ExperimentSpec, label: str, extra_overrides: dict[str, Any], extra_tags: tuple[str, ...]) -> ExperimentSpec:
        description = f"{spec.description} OOD variant: {label}."
        return ExperimentSpec(
            key=f"{label.lower().replace(' ', '_')}_{spec.key}",
            method_name=f"{spec.method_name} [{label}]",
            kind=spec.kind,
            description=description,
            config_overrides={**spec.config_overrides, **extra_overrides},
            baseline_policy=spec.baseline_policy,
            tags=(*spec.tags, "ood", *extra_tags),
        )

    if kind in {"all", "persistence"}:
        for spec in base_specs:
            specs.append(
                _with_env(
                    spec,
                    "OOD-Persistent",
                    {"regime_transition": _transition_variant(base_config.regime_transition, mode="persistent")},
                    ("persistence",),
                )
            )

    if kind in {"all", "switch"}:
        for spec in base_specs:
            specs.append(
                _with_env(
                    spec,
                    "OOD-FastSwitch",
                    {"regime_transition": _transition_variant(base_config.regime_transition, mode="fast")},
                    ("switch",),
                )
            )

    if kind in {"all", "volatility"}:
        for spec in base_specs:
            specs.append(
                _with_env(
                    spec,
                    "OOD-HighVol",
                    {"regime_params": _scaled_regime_params(base_config.regime_params, scale=1.5)},
                    ("volatility",),
                )
            )

    return SuitePlan(f"ood-{kind}", specs, base_config, "Out-of-distribution generalization sweeps.")


def build_suite(suite_name: str, base_config: TrainingConfig, ood_kind: str = "all", ablation_kind: str = "all") -> SuitePlan:
    suite_name = suite_name.lower().strip()
    if suite_name == "full":
        return _build_full_suite(base_config)
    if suite_name == "smoke":
        return _build_smoke_suite(base_config)
    if suite_name == "ablation":
        return _build_ablation_suite(base_config, kind=ablation_kind)
    if suite_name == "ood":
        return _build_ood_suite(base_config, kind=ood_kind)
    if suite_name == "all":
        full = _build_full_suite(base_config)
        ablations = _build_ablation_suite(base_config, kind=ablation_kind)
        ood = _build_ood_suite(base_config, kind=ood_kind)
        return SuitePlan("all", [*full.specs, *ablations.specs, *ood.specs], base_config, "Full benchmark + ablations + OOD sweeps.")
    raise ValueError(f"Unknown suite: {suite_name}")


def _materialize_config(base: TrainingConfig, overrides: dict[str, Any], seed: int) -> TrainingConfig:
    payload = dict(overrides)
    payload["seed"] = seed
    payload.setdefault("seeds", base.seeds)
    payload.setdefault("fixed_eval_seeds", base.fixed_eval_seeds)
    payload.setdefault("artifact_root", base.artifact_root)
    payload.setdefault("device", base.device)
    payload.setdefault("cpu_threads", base.cpu_threads)
    payload.setdefault("process_priority", base.process_priority)
    payload.setdefault("autostart", False)
    payload.setdefault("curriculum_mode", base.curriculum_mode)
    payload.setdefault("experiment_name", base.experiment_name)

    if isinstance(payload.get("agent_type"), str):
        payload["agent_type"] = AgentType(payload["agent_type"])
    if isinstance(payload.get("curriculum_mode"), str):
        payload["curriculum_mode"] = CurriculumMode(payload["curriculum_mode"])
    if isinstance(payload.get("seeds"), list):
        payload["seeds"] = tuple(payload["seeds"])
    if isinstance(payload.get("fixed_eval_seeds"), list):
        payload["fixed_eval_seeds"] = tuple(payload["fixed_eval_seeds"])
    return replace(base, **payload)


def _run_baseline_seed(config: TrainingConfig, policy_name: str, seed: int) -> SeedResult:
    step_pnls: list[float] = []
    step_regimes: list[str] = []
    eval_rewards: list[float] = []
    eval_returns: list[float] = []
    pooled_financial_metrics: list[dict[str, Any]] = []

    for eval_seed in config.fixed_eval_seeds:
        episode = _rollout_policy_episode(config, policy_name=policy_name, seed=eval_seed, policy_seed=seed)
        step_pnls.extend(episode["step_pnls"])
        step_regimes.extend(episode["step_regimes"])
        eval_rewards.append(float(episode["total_reward"]))
        eval_returns.append(float(episode["strategy_return"]))
        pooled_financial_metrics.append(episode["financial_metrics"])

    fin = episode_metrics(np.asarray(step_pnls, dtype=np.float64), step_regimes, _regime_labels())
    cumulative_return = float(np.mean(eval_returns)) if eval_returns else 0.0

    return SeedResult(
        seed=seed,
        cumulative_return=cumulative_return,
        sharpe=float(fin.get("sharpe", 0.0)),
        sortino=float(fin.get("sortino", 0.0)),
        max_drawdown=float(fin.get("max_drawdown", 0.0)),
        calmar=float(fin.get("calmar", 0.0)),
        win_rate=float(fin.get("win_rate", 0.0)),
        profit_factor=float(fin.get("profit_factor", 0.0)),
        avg_eval_reward=float(np.mean(eval_rewards)) if eval_rewards else 0.0,
        per_regime=fin.get("per_regime", {}) if isinstance(fin.get("per_regime", {}), dict) else {},
        extra={
            "policy": policy_name,
            "evalSeeds": list(config.fixed_eval_seeds),
            "pooledEpisodeMetrics": _jsonable(pooled_financial_metrics),
        },
    )


def _run_trained_seed(config: TrainingConfig, seed: int) -> tuple[str | None, SeedResult]:
    manager = TrainingManager(config)
    try:
        run_id = manager.run_new_run_blocking()
    except Exception as exc:
        return None, SeedResult(
            seed=seed,
            extra={
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
        )

    run_summary = manager.store.read_run_summary(run_id)
    checkpoint_index = manager.checkpoints(run_id) or {}
    checkpoints = checkpoint_index.get("checkpoints", [])
    checkpoint_summary = checkpoints[-1] if checkpoints else {}
    checkpoint_id = checkpoint_summary.get("checkpointId") or run_summary.get("latestCheckpointId")

    if checkpoint_id is not None:
        checkpoint_payload = manager.checkpoint_episode(checkpoint_id, run_id=run_id)
        fin_metrics = checkpoint_summary.get("financialMetrics") or (checkpoint_payload.get("financialMetrics", {}) if isinstance(checkpoint_payload, dict) else {})
    else:
        fin_metrics = {}

    if not fin_metrics:
        metrics_payload = manager.metrics(run_id) or {}
        series = metrics_payload.get("series", [])
        last_episode = series[-1] if series else {}
        fin_metrics = _normalize_financial_metrics(last_episode)

    regime_analysis = None
    if checkpoint_id is not None:
        try:
            regime_analysis = manager.store.checkpoint_regime_analysis(run_id, checkpoint_id)
        except Exception:
            regime_analysis = None

    cumulative_return = _first_numeric(
        checkpoint_summary.get("agentReturn"),
        checkpoint_summary.get("strategyReturn"),
        checkpoint_summary.get("cumulative_return"),
        fin_metrics.get("cumulative_return"),
        default=0.0,
    )

    seed_result = SeedResult(
        seed=seed,
        cumulative_return=cumulative_return,
        sharpe=float(fin_metrics.get("sharpe", 0.0)),
        sortino=float(fin_metrics.get("sortino", 0.0)),
        max_drawdown=float(fin_metrics.get("max_drawdown", fin_metrics.get("maxDrawdown", 0.0))),
        calmar=float(fin_metrics.get("calmar", 0.0)),
        win_rate=float(fin_metrics.get("win_rate", 0.0)),
        profit_factor=float(fin_metrics.get("profit_factor", 0.0)),
        avg_eval_reward=float(checkpoint_summary.get("avgEvalReward", 0.0)),
        per_regime=fin_metrics.get("per_regime", {}) if isinstance(fin_metrics.get("per_regime", {}), dict) else {},
        gate_nmi=_optional_float(checkpoint_summary.get("nmi")),
        gate_ari=_optional_float(checkpoint_summary.get("ari")),
        expert_utilization=_float_list(checkpoint_summary.get("expertUtilization")),
        extra={
            "status": run_summary.get("status", "unknown"),
            "runId": run_id,
            "checkpointId": checkpoint_id,
            "agentType": run_summary.get("agentType"),
            "financialMetrics": _normalize_financial_metrics(fin_metrics),
            "regimeAnalysis": _jsonable(regime_analysis) if regime_analysis is not None else None,
        },
    )
    return run_id, seed_result


def _rollout_policy_episode(
    config: TrainingConfig,
    policy_name: str,
    seed: int,
    policy_seed: int,
) -> dict[str, Any]:
    env = SyntheticMarketEnv(config)
    env.reset(seed=seed)
    rng = np.random.default_rng(policy_seed * 1_000_003 + seed)

    done = False
    step_pnls: list[float] = []
    step_regimes: list[str] = []
    total_reward = 0.0
    strategy_return = 0.0

    while not done:
        action = _baseline_action(policy_name, rng)
        _, reward, done, info = env.step(action)
        total_reward += float(reward)
        pnl_net = float(info["pnl"]) - float(info["transaction_cost"])
        strategy_return += pnl_net
        step_pnls.append(pnl_net)
        step_regimes.append(str(info["regime"]))

    fin = episode_metrics(np.asarray(step_pnls, dtype=np.float64), step_regimes, _regime_labels())
    return {
        "total_reward": total_reward,
        "strategy_return": strategy_return,
        "step_pnls": step_pnls,
        "step_regimes": step_regimes,
        "financial_metrics": fin,
    }


def _baseline_action(policy_name: str, rng: np.random.Generator) -> int:
    if policy_name == "buy_hold":
        return 2
    if policy_name == "random":
        return int(rng.integers(0, len(_action_labels())))
    raise ValueError(f"Unknown baseline policy: {policy_name}")


def _collect_result_for_spec(
    spec: ExperimentSpec,
    base_config: TrainingConfig,
    seeds: Sequence[int],
    dry_run: bool = False,
) -> tuple[ExperimentResult, list[ExecutionRecord]]:
    records: list[ExecutionRecord] = []
    seed_results: list[SeedResult] = []

    for seed in seeds:
        config = _materialize_config(
            base_config,
            {
                **spec.config_overrides,
                "experiment_name": f"{base_config.experiment_name}:{spec.key}",
            },
            seed,
        )

        if dry_run:
            records.append(
                ExecutionRecord(
                    spec_key=spec.key,
                    method_name=spec.method_name,
                    seed=seed,
                    status="planned",
                    result={"config": _config_snapshot(config)},
                )
            )
            continue

        if spec.kind == "baseline":
            seed_result = _run_baseline_seed(config, policy_name=spec.baseline_policy or "random", seed=seed)
            seed_result.extra.update({"status": "completed", "suite": config.experiment_name, "specKey": spec.key})
            seed_results.append(seed_result)
            records.append(
                ExecutionRecord(
                    spec_key=spec.key,
                    method_name=spec.method_name,
                    seed=seed,
                    status="completed",
                    result={"seedResult": _seed_result_to_dict(seed_result)},
                )
            )
            continue

        run_id, seed_result = _run_trained_seed(config, seed)
        seed_result.extra.update({"suite": config.experiment_name, "specKey": spec.key})
        if run_id is not None:
            seed_result.extra["runId"] = run_id
        seed_results.append(seed_result)
        records.append(
            ExecutionRecord(
                spec_key=spec.key,
                method_name=spec.method_name,
                seed=seed,
                status=seed_result.extra.get("status", "completed"),
                run_id=run_id,
                checkpoint_id=seed_result.extra.get("checkpointId"),
                result={"seedResult": _seed_result_to_dict(seed_result)},
                error=seed_result.extra.get("error"),
            )
        )

    experiment_result = ExperimentResult(method_name=spec.method_name, seed_results=_successful_seed_results(seed_results))
    return experiment_result, records


def _successful_seed_results(seed_results: Sequence[SeedResult]) -> list[SeedResult]:
    successful: list[SeedResult] = []
    for result in seed_results:
        status = str(result.extra.get("status", "completed")).lower()
        if status not in {"failed", "error"}:
            successful.append(result)
    return successful


def _config_snapshot(config: TrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["agent_type"] = config.agent_type.value
    payload["curriculum_mode"] = config.curriculum_mode.value
    payload["artifact_root"] = str(config.artifact_root)
    payload["seeds"] = list(config.seeds)
    payload["fixed_eval_seeds"] = list(config.fixed_eval_seeds)
    return _jsonable(payload)


def _normalize_financial_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized = _jsonable(metrics) if isinstance(metrics, dict) else {}
    if not isinstance(normalized, dict):
        return {}
    if "maxDrawdown" in normalized and "max_drawdown" not in normalized:
        normalized["max_drawdown"] = normalized["maxDrawdown"]
    if "avgEvalReward" in normalized and "avg_eval_reward" not in normalized:
        normalized["avg_eval_reward"] = normalized["avgEvalReward"]
    return normalized


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_numeric(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    return None


def _regime_labels() -> tuple[str, ...]:
    return REGIME_LABELS


def _action_labels() -> tuple[str, ...]:
    return ACTION_LABELS


def _transition_variant(
    transition: tuple[tuple[float, ...], ...],
    mode: str,
    strength: float = 0.18,
) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(transition, dtype=np.float64)
    n = matrix.shape[0]
    out = matrix.copy()
    for row_idx in range(n):
        row = out[row_idx]
        if mode == "persistent":
            row = row * (1.0 - strength)
            row[row_idx] += strength
        elif mode == "fast":
            uniform = np.full_like(row, 1.0 / n)
            row = row * (1.0 - strength) + uniform * strength
        else:
            raise ValueError(f"Unknown transition mode: {mode}")
        row = np.clip(row, 1e-8, None)
        row = row / row.sum()
        out[row_idx] = row
    return tuple(tuple(float(v) for v in row) for row in out)


def _scaled_regime_params(regime_params: dict[str, dict[str, float]], scale: float) -> dict[str, dict[str, float]]:
    params = copy.deepcopy(regime_params)
    for label, values in params.items():
        if "vol" in values:
            values["vol"] = float(values["vol"]) * scale
        if "jump_scale" in values:
            values["jump_scale"] = float(values["jump_scale"]) * scale
    return params


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            pass
    return str(value)


def _seed_result_to_dict(result: SeedResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "cumulative_return": result.cumulative_return,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "max_drawdown": result.max_drawdown,
        "calmar": result.calmar,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "avg_eval_reward": result.avg_eval_reward,
        "per_regime": _jsonable(result.per_regime),
        "gate_nmi": result.gate_nmi,
        "gate_ari": result.gate_ari,
        "expert_utilization": result.expert_utilization,
        "extra": _jsonable(result.extra),
    }


def _experiment_result_to_dict(result: ExperimentResult) -> dict[str, Any]:
    return {
        "methodName": result.method_name,
        "nSeeds": len(result.seed_results),
        "mean": {
            "cumulative_return": result.mean("cumulative_return") if result.seed_results else 0.0,
            "sharpe": result.mean("sharpe") if result.seed_results else 0.0,
            "sortino": result.mean("sortino") if result.seed_results else 0.0,
            "max_drawdown": result.mean("max_drawdown") if result.seed_results else 0.0,
            "calmar": result.mean("calmar") if result.seed_results else 0.0,
            "win_rate": result.mean("win_rate") if result.seed_results else 0.0,
            "profit_factor": result.mean("profit_factor") if result.seed_results else 0.0,
            "avg_eval_reward": result.mean("avg_eval_reward") if result.seed_results else 0.0,
            "gate_nmi": _mean_optional(result, "gate_nmi"),
            "gate_ari": _mean_optional(result, "gate_ari"),
        },
        "ci95": {
            metric: list(result.ci_95(metric)) if result.seed_results else [0.0, 0.0]
            for metric in ("cumulative_return", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "profit_factor", "avg_eval_reward")
        },
        "seedResults": [_seed_result_to_dict(seed_result) for seed_result in result.seed_results],
    }


def _execution_record_to_dict(record: ExecutionRecord) -> dict[str, Any]:
    return {
        "specKey": record.spec_key,
        "methodName": record.method_name,
        "seed": record.seed,
        "status": record.status,
        "runId": record.run_id,
        "checkpointId": record.checkpoint_id,
        "result": _jsonable(record.result),
        "error": _jsonable(record.error),
    }


def _mean_optional(result: ExperimentResult, attr: str) -> float:
    values = [getattr(item, attr) for item in result.seed_results if getattr(item, attr) is not None]
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _render_console_plan(plan: SuitePlan) -> str:
    lines = [
        f"Suite: {plan.name}",
        f"Description: {plan.description}",
        f"Methods: {len(plan.specs)}",
        f"Seeds per method: {len(plan.base_config.seeds)}",
        f"Eval seeds per run: {len(plan.base_config.fixed_eval_seeds)}",
        f"Estimated runs: {plan.total_runs}",
        "",
        "Specs:",
    ]
    for spec in plan.specs:
        lines.append(f"- {spec.method_name} [{spec.kind}] - {spec.description}")
    return "\n".join(lines)


def _results_markdown_table(experiment_results: dict[str, ExperimentResult]) -> str:
    headers = ["Method", "Return", "Sharpe", "Sortino", "MDD", "Eval Reward", "NMI", "ARI"]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, result in experiment_results.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{result.mean('cumulative_return'):+.4f}" if result.seed_results else "+0.0000",
                    f"{result.mean('sharpe'):+.4f}" if result.seed_results else "+0.0000",
                    f"{result.mean('sortino'):+.4f}" if result.seed_results else "+0.0000",
                    f"{result.mean('max_drawdown'):+.4f}" if result.seed_results else "+0.0000",
                    f"{result.mean('avg_eval_reward'):+.4f}" if result.seed_results else "+0.0000",
                    f"{_mean_optional(result, 'gate_nmi'):+.4f}",
                    f"{_mean_optional(result, 'gate_ari'):+.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _render_latex_report(experiment_results: dict[str, ExperimentResult]) -> str:
    if not experiment_results:
        return "% No results available"
    try:
        return results_to_latex(
            experiment_results,
            metrics=("cumulative_return", "sharpe", "max_drawdown", "avg_eval_reward"),
            metric_labels=("Return", "Sharpe", "MDD", "EvalReward"),
        )
    except TypeError:
        return results_to_latex(experiment_results)


def _render_markdown_report(
    suite: SuitePlan,
    experiment_results: dict[str, ExperimentResult],
    execution_records: list[ExecutionRecord],
) -> str:
    lines: list[str] = []
    lines.append("# Regime Lens Experiment Report")
    lines.append("")
    lines.append(f"- Suite: `{suite.name}`")
    lines.append(f"- Description: {suite.description}")
    lines.append(f"- Generated at: `{datetime.now(tz=UTC).isoformat()}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(_results_markdown_table(experiment_results))
    lines.append("")
    if any(record.status == "failed" for record in execution_records):
        lines.append("## Failures")
        lines.append("")
        for record in execution_records:
            if record.status == "failed":
                lines.append(f"- `{record.method_name}` seed `{record.seed}` failed: `{record.error}`")
        lines.append("")
    baseline_key = next((name for name in experiment_results if name.startswith("Vanilla DQN")), None)
    baseline = experiment_results.get(baseline_key) if baseline_key is not None else None
    if baseline is not None:
        lines.append("## Delta vs Vanilla DQN")
        lines.append("")
        for name, result in experiment_results.items():
            if name == "Vanilla DQN":
                continue
            try:
                stats = result.vs_baseline(baseline, attr="cumulative_return")
                lines.append(
                    f"- `{name}`: t={stats['t_statistic']:+.3f}, p={stats['p_value']:.4g}, d={stats['effect_size_cohens_d']:+.3f}"
                )
            except Exception:
                continue
        lines.append("")
    lines.append("## LaTeX")
    lines.append("")
    lines.append("```latex")
    lines.append(_render_latex_report(experiment_results))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _suite_report(
    suite: SuitePlan,
    experiment_results: dict[str, ExperimentResult],
    execution_records: list[ExecutionRecord],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generatedAt": datetime.now(tz=UTC).isoformat(),
        "suite": suite.name,
        "description": suite.description,
        "baseConfig": _config_snapshot(suite.base_config),
        "specs": [spec.to_manifest() for spec in suite.specs],
        "results": {method: _experiment_result_to_dict(result) for method, result in experiment_results.items()},
        "executionRecords": [_execution_record_to_dict(record) for record in execution_records],
    }

    (output_dir / "report.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=True), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_markdown_report(suite, experiment_results, execution_records), encoding="utf-8")
    (output_dir / "results.tex").write_text(_render_latex_report(experiment_results), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generatedAt": report["generatedAt"],
                "suite": suite.name,
                "description": suite.description,
                "baseConfig": _config_snapshot(suite.base_config),
                "specs": [spec.to_manifest() for spec in suite.specs],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "execution_records.json").write_text(
        json.dumps([_execution_record_to_dict(record) for record in execution_records], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "reportPath": str(output_dir / "report.json"),
        "markdownPath": str(output_dir / "report.md"),
        "latexPath": str(output_dir / "results.tex"),
        "manifestPath": str(output_dir / "manifest.json"),
        "executionRecordsPath": str(output_dir / "execution_records.json"),
    }


def _config_from_snapshot(snapshot: dict[str, Any]) -> TrainingConfig:
    payload = dict(snapshot)
    payload["artifact_root"] = Path(payload.get("artifact_root", TrainingConfig().artifact_root))
    if isinstance(payload.get("agent_type"), str):
        payload["agent_type"] = AgentType(payload["agent_type"])
    if isinstance(payload.get("curriculum_mode"), str):
        payload["curriculum_mode"] = CurriculumMode(payload["curriculum_mode"])
    if isinstance(payload.get("seeds"), list):
        payload["seeds"] = tuple(int(x) for x in payload["seeds"])
    if isinstance(payload.get("fixed_eval_seeds"), list):
        payload["fixed_eval_seeds"] = tuple(int(x) for x in payload["fixed_eval_seeds"])
    allowed = {k: v for k, v in payload.items() if k in TrainingConfig.__dataclass_fields__}
    return TrainingConfig(**allowed)


def _load_records_for_spec(output_root: Path, spec: ExperimentSpec) -> list[ExecutionRecord]:
    records_path = output_root / "execution_records.json"
    if not records_path.exists():
        return []
    data = json.loads(records_path.read_text(encoding="utf-8"))
    records: list[ExecutionRecord] = []
    for item in data:
        if item.get("specKey") != spec.key:
            continue
        records.append(
            ExecutionRecord(
                spec_key=str(item.get("specKey")),
                method_name=str(item.get("methodName")),
                seed=int(item.get("seed", 0)),
                status=str(item.get("status", "unknown")),
                run_id=item.get("runId"),
                checkpoint_id=item.get("checkpointId"),
                result=item.get("result", {}),
                error=item.get("error"),
            )
        )
    return records


def _record_to_seed_result(record: ExecutionRecord) -> SeedResult:
    seed_payload = record.result.get("seedResult", {})
    return SeedResult(
        seed=int(seed_payload.get("seed", record.seed)),
        cumulative_return=float(seed_payload.get("cumulative_return", 0.0)),
        sharpe=float(seed_payload.get("sharpe", 0.0)),
        sortino=float(seed_payload.get("sortino", 0.0)),
        max_drawdown=float(seed_payload.get("max_drawdown", 0.0)),
        calmar=float(seed_payload.get("calmar", 0.0)),
        win_rate=float(seed_payload.get("win_rate", 0.0)),
        profit_factor=float(seed_payload.get("profit_factor", 0.0)),
        avg_eval_reward=float(seed_payload.get("avg_eval_reward", 0.0)),
        per_regime=seed_payload.get("per_regime", {}) if isinstance(seed_payload.get("per_regime", {}), dict) else {},
        gate_nmi=_optional_float(seed_payload.get("gate_nmi")),
        gate_ari=_optional_float(seed_payload.get("gate_ari")),
        expert_utilization=_float_list(seed_payload.get("expert_utilization")),
        extra=seed_payload.get("extra", {}),
    )


def _result_from_dict(payload: dict[str, Any]) -> ExperimentResult:
    seed_results = [
        SeedResult(
            seed=int(item.get("seed", 0)),
            cumulative_return=float(item.get("cumulative_return", 0.0)),
            sharpe=float(item.get("sharpe", 0.0)),
            sortino=float(item.get("sortino", 0.0)),
            max_drawdown=float(item.get("max_drawdown", 0.0)),
            calmar=float(item.get("calmar", 0.0)),
            win_rate=float(item.get("win_rate", 0.0)),
            profit_factor=float(item.get("profit_factor", 0.0)),
            avg_eval_reward=float(item.get("avg_eval_reward", 0.0)),
            per_regime=item.get("per_regime", {}) if isinstance(item.get("per_regime", {}), dict) else {},
            gate_nmi=_optional_float(item.get("gate_nmi")),
            gate_ari=_optional_float(item.get("gate_ari")),
            expert_utilization=_float_list(item.get("expert_utilization")),
            extra=item.get("extra", {}),
        )
        for item in payload.get("seedResults", [])
    ]
    return ExperimentResult(method_name=str(payload.get("methodName", "unknown")), seed_results=seed_results)


def _execution_record_from_dict(payload: dict[str, Any]) -> ExecutionRecord:
    return ExecutionRecord(
        spec_key=str(payload.get("specKey", "")),
        method_name=str(payload.get("methodName", "")),
        seed=int(payload.get("seed", 0)),
        status=str(payload.get("status", "unknown")),
        run_id=payload.get("runId"),
        checkpoint_id=payload.get("checkpointId"),
        result=payload.get("result", {}),
        error=payload.get("error"),
    )


def plan_experiments(
    suite_name: str,
    base_config: TrainingConfig,
    ood_kind: str = "all",
    ablation_kind: str = "all",
) -> SuitePlan:
    return build_suite(suite_name, base_config, ood_kind=ood_kind, ablation_kind=ablation_kind)


def run_experiments(
    suite_name: str,
    base_config: TrainingConfig,
    *,
    output_root: Path | None = None,
    ood_kind: str = "all",
    ablation_kind: str = "all",
    dry_run: bool = False,
) -> dict[str, Any]:
    suite = build_suite(suite_name, base_config, ood_kind=ood_kind, ablation_kind=ablation_kind)
    if output_root is None:
        output_root = base_config.artifact_root / "_experiments" / suite.name
    output_root = output_root / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

    experiment_results: dict[str, ExperimentResult] = {}
    execution_records: list[ExecutionRecord] = []

    for spec in suite.specs:
        result, records = _collect_result_for_spec(spec, suite.base_config, suite.base_config.seeds, dry_run=dry_run)
        execution_records.extend(records)
        if not dry_run:
            experiment_results[spec.method_name] = result

    outputs: dict[str, str] = {}
    if not dry_run:
        outputs = _suite_report(suite, experiment_results, execution_records, output_root)

    return {
        "suite": suite.name,
        "description": suite.description,
        "baseConfig": _config_snapshot(suite.base_config),
        "specs": [spec.to_manifest() for spec in suite.specs],
        "experimentResults": {name: _experiment_result_to_dict(result) for name, result in experiment_results.items()},
        "executionRecords": [_execution_record_to_dict(record) for record in execution_records],
        "outputs": outputs,
    }


def report_experiments(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite_name = str(manifest.get("suite", "unknown"))
    base_config = _config_from_snapshot(manifest.get("baseConfig", {}))
    suite = build_suite(suite_name, base_config)
    output_root = manifest_path.parent

    experiment_results: dict[str, ExperimentResult] = {}
    execution_records: list[ExecutionRecord] = []

    for spec in suite.specs:
        records = _load_records_for_spec(output_root, spec)
        execution_records.extend(records)
        seed_results = [_record_to_seed_result(record) for record in records if record.status == "completed" and record.result.get("seedResult")]
        if seed_results:
            experiment_results[spec.method_name] = ExperimentResult(method_name=spec.method_name, seed_results=seed_results)

    outputs = _suite_report(suite, experiment_results, execution_records, output_root)
    return {
        "suite": suite.name,
        "manifestPath": str(manifest_path),
        "outputs": outputs,
        "experimentResults": {name: _experiment_result_to_dict(result) for name, result in experiment_results.items()},
    }


def _build_base_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig(experiment_name=args.experiment_name)
    updates: dict[str, Any] = {}

    if args.artifact_root is not None:
        updates["artifact_root"] = args.artifact_root
    if args.seeds is not None:
        updates["seeds"] = _parse_int_tuple(args.seeds)
    if args.eval_seeds is not None:
        updates["fixed_eval_seeds"] = _parse_int_tuple(args.eval_seeds)
    if args.episodes is not None:
        updates["episodes"] = int(args.episodes)
    if args.checkpoint_interval is not None:
        updates["checkpoint_interval"] = int(args.checkpoint_interval)
    if args.metrics_flush_interval is not None:
        updates["metrics_flush_interval"] = int(args.metrics_flush_interval)
    if args.evaluation_episodes is not None:
        updates["evaluation_episodes"] = int(args.evaluation_episodes)
    if args.device is not None:
        updates["device"] = args.device
    if args.cpu_threads is not None:
        updates["cpu_threads"] = int(args.cpu_threads)
    if args.process_priority is not None:
        updates["process_priority"] = args.process_priority

    return replace(config, **updates)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(value) for value in values)


def _result_from_experiment_dict(payload: dict[str, Any]) -> ExperimentResult:
    return _result_from_dict(payload)


def _execution_record_from_dict(payload: dict[str, Any]) -> ExecutionRecord:
    return ExecutionRecord(
        spec_key=str(payload.get("specKey", "")),
        method_name=str(payload.get("methodName", "")),
        seed=int(payload.get("seed", 0)),
        status=str(payload.get("status", "unknown")),
        run_id=payload.get("runId"),
        checkpoint_id=payload.get("checkpointId"),
        result=payload.get("result", {}),
        error=payload.get("error"),
    )


def _cmd_plan(args: argparse.Namespace) -> int:
    base_config = _build_base_config(args)
    plan = plan_experiments(args.suite, base_config, ood_kind=args.ood_kind, ablation_kind=args.ablation_kind)
    print(_render_console_plan(plan))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    base_config = _build_base_config(args)
    result = run_experiments(
        args.suite,
        base_config,
        output_root=base_config.artifact_root / "_experiments" / args.experiment_name,
        ood_kind=args.ood_kind,
        ablation_kind=args.ablation_kind,
        dry_run=getattr(args, "dry_run", False),
    )
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    print(
        _render_console_summary(
            build_suite(args.suite, base_config, ood_kind=args.ood_kind, ablation_kind=args.ablation_kind),
            {name: _result_from_experiment_dict(value) for name, value in result.get("experimentResults", {}).items()},
            [_execution_record_from_dict(item) for item in result.get("executionRecords", [])],
            result.get("outputs", {}),
        )
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    result = report_experiments(Path(args.manifest))
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, ensure_ascii=True))
    else:
        print(json.dumps(result["outputs"], indent=2, ensure_ascii=True))
    return 0


def _render_console_summary(
    suite: SuitePlan,
    experiment_results: dict[str, ExperimentResult],
    records: list[ExecutionRecord],
    output_paths: dict[str, str] | None = None,
) -> str:
    lines = [
        f"Completed suite: {suite.name}",
        f"Methods: {len(experiment_results)}",
        f"Records: {len(records)}",
        "",
        _results_markdown_table(experiment_results),
    ]
    if output_paths:
        lines.extend(["", "Outputs:"])
        for key, value in output_paths.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="smoke", choices=("smoke", "full", "ablation", "ood", "all"), help="Which suite to plan or run.")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Override the artifact root.")
    parser.add_argument("--experiment-name", default="regime_lens_experiments", help="Base experiment label written to artifacts.")
    parser.add_argument("--seeds", default=None, help="Comma-separated training seeds. Defaults to the config seeds.")
    parser.add_argument("--eval-seeds", default=None, help="Comma-separated evaluation seeds. Defaults to the config eval seeds.")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes.")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Override checkpoint interval.")
    parser.add_argument("--metrics-flush-interval", type=int, default=None, help="Override metrics flush interval.")
    parser.add_argument("--evaluation-episodes", type=int, default=None, help="Override evaluation episodes per checkpoint.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU thread override.")
    parser.add_argument("--process-priority", default=None, help="Process priority hint.")
    parser.add_argument("--ood-kind", default="all", choices=("all", "persistence", "switch", "volatility"), help="Which OOD subset to use.")
    parser.add_argument("--ablation-kind", default="all", choices=("all", "experts", "gate", "lb", "hidden"), help="Which ablation subset to use.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regime Lens experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Print the selected suite without running it.")
    _add_common_args(plan_parser)

    run_parser = subparsers.add_parser("run", help="Run the selected suite and write a report bundle.")
    _add_common_args(run_parser)
    run_parser.add_argument("--dry-run", action="store_true", help="Build the plan and artifacts but skip execution.")
    run_parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output.")

    report_parser = subparsers.add_parser("report", help="Rebuild a report from an existing manifest.")
    report_parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json created by this runner.")
    report_parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output.")

    args = parser.parse_args(argv)
    command = args.command or "plan"

    if command == "plan":
        return _cmd_plan(args)
    if command == "run":
        return _cmd_run(args)
    if command == "report":
        return _cmd_report(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
