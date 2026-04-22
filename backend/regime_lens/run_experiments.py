"""Experiment runner for Regime Lens / RCMoE research runs."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from backend.regime_lens.config import (
        ACTION_LABELS,
        REGIME_LABELS,
        AgentType,
        AlgorithmType,
        CurriculumMode,
        DataSource,
        GateType,
        TrackingBackend,
        TrainingConfig,
        config_from_snapshot,
        config_to_snapshot,
    )
    from backend.regime_lens.continuous_market import make_market_env
    from backend.regime_lens.config_io import load_training_config
    from backend.regime_lens.experiment import (
        ExperimentResult,
        SeedResult,
        compare_methods_robustly,
        results_to_latex,
    )
    from backend.regime_lens.market import SyntheticMarketEnv
    from backend.regime_lens.metrics import episode_metrics
    from backend.regime_lens.training import TrainingManager
else:
    from .config import (
        ACTION_LABELS,
        REGIME_LABELS,
        AgentType,
        AlgorithmType,
        CurriculumMode,
        DataSource,
        GateType,
        TrackingBackend,
        TrainingConfig,
        config_from_snapshot,
        config_to_snapshot,
    )
    from .continuous_market import make_market_env
    from .config_io import load_training_config
    from .experiment import (
        ExperimentResult,
        SeedResult,
        compare_methods_robustly,
        results_to_latex,
    )
    from .market import SyntheticMarketEnv
    from .metrics import episode_metrics
    from .training import TrainingManager


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
        return ExperimentSpec(
            key=f"{label.lower().replace(' ', '_')}_{spec.key}",
            method_name=f"{spec.method_name} [{label}]",
            kind=spec.kind,
            description=f"{spec.description} OOD variant: {label}.",
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

    if kind in {"all", "drift"}:
        for spec in base_specs:
            specs.append(
                _with_env(
                    spec,
                    "OOD-Drift",
                    {
                        "nonstationary_mode": "cyclical",
                        "nonstationary_drift_scale": max(float(base_config.nonstationary_drift_scale), 0.35),
                    },
                    ("drift",),
                )
            )

    return SuitePlan(f"ood-{kind}", specs, base_config, "Out-of-distribution generalization sweeps.")


def build_suite(
    suite_name: str,
    base_config: TrainingConfig,
    ood_kind: str = "all",
    ablation_kind: str = "all",
) -> SuitePlan:
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
        return SuitePlan(
            "all",
            [*full.specs, *ablations.specs, *ood.specs],
            base_config,
            "Full benchmark + ablations + OOD sweeps.",
        )
    raise ValueError(f"Unknown suite: {suite_name}")


def _spec_from_manifest(payload: dict[str, Any]) -> ExperimentSpec:
    return ExperimentSpec(
        key=str(payload.get("key", "")),
        method_name=str(payload.get("methodName", "")),
        kind=str(payload.get("kind", "trained")),
        description=str(payload.get("description", "")),
        config_overrides=payload.get("configOverrides", {}) if isinstance(payload.get("configOverrides", {}), dict) else {},
        baseline_policy=payload.get("baselinePolicy"),
        tags=tuple(str(item) for item in payload.get("tags", [])),
    )


def _materialize_config(base: TrainingConfig, overrides: dict[str, Any], seed: int) -> TrainingConfig:
    snapshot = config_to_snapshot(base)
    snapshot.update(_jsonable(overrides))
    snapshot["seed"] = int(seed)
    return config_from_snapshot(snapshot)


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


def _run_trained_seed(
    config: TrainingConfig,
    seed: int,
    *,
    source_artifact_root: Path | None = None,
) -> tuple[str | None, SeedResult]:
    source_root = source_artifact_root or config.artifact_root
    original_resume_run = config.resume_run_id
    original_resume_checkpoint = config.resume_checkpoint_id
    staged_config, staged_resume = _stage_resume_source(config, source_root)
    manager = TrainingManager(staged_config)

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

    if staged_resume is not None:
        corrected_config = replace(
            staged_config,
            resume_run_id=original_resume_run,
            resume_checkpoint_id=staged_resume["originalCheckpointId"],
        )
        manager.store.update_run_summary(
            run_id,
            {
                "config": config_to_snapshot(corrected_config),
                "resumedFromRunId": original_resume_run,
                "resumedFromCheckpointId": staged_resume["originalCheckpointId"],
            },
        )

    run_summary = manager.store.read_run_summary(run_id)
    checkpoint_index = manager.checkpoints(run_id) or {}
    checkpoints = checkpoint_index.get("checkpoints", [])
    checkpoint_summary = checkpoints[-1] if checkpoints else {}
    checkpoint_id = checkpoint_summary.get("checkpointId") or run_summary.get("latestCheckpointId")

    if checkpoint_id is not None:
        checkpoint_payload = manager.checkpoint_episode(checkpoint_id, run_id=run_id)
        fin_metrics = checkpoint_summary.get("financialMetrics") or (
            checkpoint_payload.get("financialMetrics", {}) if isinstance(checkpoint_payload, dict) else {}
        )
    else:
        fin_metrics = {}

    if not fin_metrics:
        metrics_payload = manager.metrics(run_id) or {}
        series = metrics_payload.get("series", [])
        last_episode = series[-1] if series else {}
        fin_metrics = _normalize_financial_metrics(last_episode)

    regime_analysis = None
    stats_payload = None
    explainability_payload = None
    data_fit_payload = None
    if checkpoint_id is not None:
        try:
            regime_analysis = manager.store.checkpoint_regime_analysis(run_id, checkpoint_id)
        except Exception:
            regime_analysis = None
        try:
            stats_payload = manager.store.checkpoint_stats(run_id, checkpoint_id)
        except Exception:
            stats_payload = None
        try:
            explainability_payload = manager.store.checkpoint_explainability(run_id, checkpoint_id)
        except Exception:
            explainability_payload = None
        try:
            data_fit_payload = manager.store.checkpoint_data_fit(run_id, checkpoint_id)
        except Exception:
            data_fit_payload = None

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
            "artifactRoot": str(staged_config.artifact_root),
            "financialMetrics": _normalize_financial_metrics(fin_metrics),
            "stats": _jsonable(stats_payload),
            "explainability": _jsonable(explainability_payload),
            "dataFit": _jsonable(data_fit_payload),
            "regimeAnalysis": _jsonable(regime_analysis) if regime_analysis is not None else None,
        },
    )
    return run_id, seed_result


def _stage_resume_source(
    config: TrainingConfig,
    source_artifact_root: Path,
) -> tuple[TrainingConfig, dict[str, str] | None]:
    if config.resume_run_id is None:
        return config, None
    if config.artifact_root.resolve() == source_artifact_root.resolve():
        return config, None

    source_run_dir = _resolve_run_dir(source_artifact_root, config.resume_run_id)
    if source_run_dir is None:
        raise FileNotFoundError(
            f"Could not locate resume run {config.resume_run_id!r} under {source_artifact_root}."
        )
    resolved_checkpoint = config.resume_checkpoint_id or _latest_checkpoint_id(source_run_dir)
    alias = f"__resume__{config.resume_run_id}"
    target_dir = config.artifact_root / alias
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_run_dir, target_dir)
    return (
        replace(config, resume_run_id=alias, resume_checkpoint_id=resolved_checkpoint),
        {
            "originalRunId": str(config.resume_run_id),
            "originalCheckpointId": str(resolved_checkpoint) if resolved_checkpoint is not None else "",
            "stagedRunId": alias,
        },
    )


def _resolve_run_dir(root: Path, run_id: str) -> Path | None:
    direct = root / run_id
    if direct.exists():
        return direct
    for path in root.rglob(run_id):
        if path.is_dir() and path.name == run_id:
            return path
    return None


def _latest_checkpoint_id(run_dir: Path) -> str | None:
    index_path = run_dir / "checkpoints" / "index.json"
    if not index_path.exists():
        return None
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints", [])
    if not checkpoints:
        return None
    return str(checkpoints[-1].get("checkpointId"))


def _rollout_policy_episode(
    config: TrainingConfig,
    policy_name: str,
    seed: int,
    policy_seed: int,
) -> dict[str, Any]:
    continuous = _uses_continuous_config(config)
    env = (
        make_market_env(
            config,
            continuous=True,
            multi_asset=bool(config.real_data_symbols and len(tuple(config.real_data_symbols)) > 1),
            asset_names=tuple(config.real_data_symbols) if config.real_data_symbols else ("asset",),
        )
        if continuous
        else SyntheticMarketEnv(config)
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(policy_seed * 1_000_003 + seed)

    done = False
    step_pnls: list[float] = []
    step_regimes: list[str] = []
    total_reward = 0.0
    strategy_return = 0.0

    while not done:
        action = _baseline_action(
            policy_name,
            rng,
            continuous=continuous,
            action_dim=int(getattr(env, "action_dim", 1)),
        )
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


def _baseline_action(
    policy_name: str,
    rng: np.random.Generator,
    *,
    continuous: bool,
    action_dim: int,
) -> int | np.ndarray:
    if continuous:
        if policy_name == "buy_hold":
            return np.ones(action_dim, dtype=np.float32)
        if policy_name == "random":
            return rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        raise ValueError(f"Unknown continuous baseline policy: {policy_name}")
    if policy_name == "buy_hold":
        return 2
    if policy_name == "random":
        return int(rng.integers(0, len(_action_labels())))
    raise ValueError(f"Unknown baseline policy: {policy_name}")


def _uses_continuous_config(config: TrainingConfig) -> bool:
    return bool(config.continuous_actions or config.algorithm != AlgorithmType.DQN)


def _build_jobs(
    suite: SuitePlan,
    output_root: Path,
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    use_isolated_roots = suite.base_config.parallel_workers > 1
    for job_index, spec in enumerate(suite.specs):
        for seed in suite.base_config.seeds:
            artifact_root = (
                output_root / "_worker_runs" / spec.key / f"seed-{seed}"
                if use_isolated_roots
                else suite.base_config.artifact_root
            )
            jobs.append(
                {
                    "index": len(jobs),
                    "suiteName": suite.name,
                    "spec": spec.to_manifest(),
                    "seed": int(seed),
                    "dryRun": dry_run,
                    "baseConfig": config_to_snapshot(suite.base_config),
                    "artifactRoot": str(artifact_root),
                    "sourceArtifactRoot": str(suite.base_config.artifact_root),
                }
            )
    return jobs


def _execute_seed_job(job: dict[str, Any]) -> dict[str, Any]:
    spec = _spec_from_manifest(job["spec"])
    base_config = config_from_snapshot(job["baseConfig"])
    seed = int(job["seed"])
    dry_run = bool(job["dryRun"])
    config = _materialize_config(
        base_config,
        {
            **spec.config_overrides,
            "artifact_root": Path(job["artifactRoot"]),
            "autostart": False,
            "experiment_name": f"{base_config.experiment_name}:{spec.key}",
        },
        seed,
    )

    if dry_run:
        record = ExecutionRecord(
            spec_key=spec.key,
            method_name=spec.method_name,
            seed=seed,
            status="planned",
            result={
                "config": _config_snapshot(config),
                "artifactRoot": str(config.artifact_root),
            },
        )
        return {
            "index": int(job["index"]),
            "specKey": spec.key,
            "methodName": spec.method_name,
            "record": _execution_record_to_dict(record),
            "seedResult": None,
        }

    if spec.kind == "baseline":
        seed_result = _run_baseline_seed(config, policy_name=spec.baseline_policy or "random", seed=seed)
        seed_result.extra.update(
            {
                "status": "completed",
                "suite": config.experiment_name,
                "specKey": spec.key,
                "artifactRoot": str(config.artifact_root),
            }
        )
        record = ExecutionRecord(
            spec_key=spec.key,
            method_name=spec.method_name,
            seed=seed,
            status="completed",
            result={"seedResult": _seed_result_to_dict(seed_result)},
        )
        return {
            "index": int(job["index"]),
            "specKey": spec.key,
            "methodName": spec.method_name,
            "record": _execution_record_to_dict(record),
            "seedResult": _seed_result_to_dict(seed_result),
        }

    run_id, seed_result = _run_trained_seed(
        config,
        seed,
        source_artifact_root=Path(job["sourceArtifactRoot"]),
    )
    seed_result.extra.update(
        {
            "suite": config.experiment_name,
            "specKey": spec.key,
        }
    )
    if run_id is not None:
        seed_result.extra["runId"] = run_id
    record = ExecutionRecord(
        spec_key=spec.key,
        method_name=spec.method_name,
        seed=seed,
        status=str(seed_result.extra.get("status", "completed")),
        run_id=run_id,
        checkpoint_id=seed_result.extra.get("checkpointId"),
        result={"seedResult": _seed_result_to_dict(seed_result)},
        error=seed_result.extra.get("error"),
    )
    return {
        "index": int(job["index"]),
        "specKey": spec.key,
        "methodName": spec.method_name,
        "record": _execution_record_to_dict(record),
        "seedResult": _seed_result_to_dict(seed_result),
    }


def _execute_jobs(jobs: list[dict[str, Any]], parallel_workers: int) -> list[dict[str, Any]]:
    if parallel_workers <= 1 or len(jobs) <= 1:
        return [_execute_seed_job(job) for job in jobs]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(parallel_workers, len(jobs))) as executor:
        future_map = {executor.submit(_execute_seed_job, job): job for job in jobs}
        for future in as_completed(future_map):
            results.append(future.result())
    results.sort(key=lambda item: int(item["index"]))
    return results


def _seed_results_from_job_payloads(
    suite: SuitePlan,
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, ExperimentResult], list[ExecutionRecord]]:
    records: list[ExecutionRecord] = []
    grouped: dict[str, list[SeedResult]] = {spec.key: [] for spec in suite.specs}
    method_names = {spec.key: spec.method_name for spec in suite.specs}

    for payload in payloads:
        record = _execution_record_from_dict(payload["record"])
        records.append(record)
        seed_payload = payload.get("seedResult")
        if seed_payload is None:
            continue
        seed_result = _record_to_seed_result(record)
        if str(seed_result.extra.get("status", "completed")).lower() not in {"failed", "error"}:
            grouped[str(payload["specKey"])].append(seed_result)

    experiment_results: dict[str, ExperimentResult] = {}
    for spec in suite.specs:
        experiment_results[spec.method_name] = ExperimentResult(
            method_name=method_names[spec.key],
            seed_results=grouped.get(spec.key, []),
        )
    return experiment_results, records


def _config_snapshot(config: TrainingConfig) -> dict[str, Any]:
    return _jsonable(config_to_snapshot(config))


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
    out = matrix.copy()
    n_rows = matrix.shape[0]
    for row_idx in range(n_rows):
        row = out[row_idx]
        if mode == "persistent":
            row = row * (1.0 - strength)
            row[row_idx] += strength
        elif mode == "fast":
            uniform = np.full_like(row, 1.0 / len(row))
            row = row * (1.0 - strength) + uniform * strength
        else:
            raise ValueError(f"Unknown transition mode: {mode}")
        row = np.clip(row, 1e-8, None)
        row = row / row.sum()
        out[row_idx] = row
    return tuple(tuple(float(value) for value in row) for row in out)


def _scaled_regime_params(regime_params: dict[str, dict[str, float]], scale: float) -> dict[str, dict[str, float]]:
    params = copy.deepcopy(regime_params)
    for values in params.values():
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
        return {str(key): _jsonable(item) for key, item in value.items()}
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
    metrics = (
        "cumulative_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "profit_factor",
        "avg_eval_reward",
    )
    return {
        "methodName": result.method_name,
        "nSeeds": len(result.seed_results),
        "mean": {
            metric: result.mean(metric) if result.seed_results else 0.0
            for metric in metrics
        }
        | {
            "gate_nmi": _mean_optional(result, "gate_nmi"),
            "gate_ari": _mean_optional(result, "gate_ari"),
        },
        "ci95": {
            metric: list(result.ci_95(metric)) if result.seed_results else [0.0, 0.0]
            for metric in metrics
        },
        "bootstrapCi95": {
            metric: list(result.bootstrap_ci(metric)) if result.seed_results else [0.0, 0.0]
            for metric in metrics
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


def _build_report_statistics(experiment_results: dict[str, ExperimentResult]) -> dict[str, Any]:
    populated = {
        method: result
        for method, result in experiment_results.items()
        if result.seed_results
    }
    baseline_name = next((name for name in populated if name.startswith("Vanilla DQN")), None)
    baseline = populated.get(baseline_name) if baseline_name is not None else None
    comparisons: dict[str, Any] = {}
    if baseline is not None:
        for name, result in populated.items():
            if name == baseline_name:
                continue
            comparisons[name] = {
                "welch": result.vs_baseline(baseline, attr="cumulative_return"),
                "robust": result.robust_vs_baseline(baseline, attr="cumulative_return"),
            }

    overall = compare_methods_robustly(populated, attr="cumulative_return") if populated else {}
    return _jsonable(
        {
            "baselineMethod": baseline_name,
            "overall": overall,
            "vsBaseline": comparisons,
        }
    )


def _render_console_plan(plan: SuitePlan) -> str:
    lines = [
        f"Suite: {plan.name}",
        f"Description: {plan.description}",
        f"Methods: {len(plan.specs)}",
        f"Seeds per method: {len(plan.base_config.seeds)}",
        f"Eval seeds per run: {len(plan.base_config.fixed_eval_seeds)}",
        f"Parallel workers: {plan.base_config.parallel_workers}",
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
    statistics: dict[str, Any],
) -> str:
    lines = [
        "# Regime Lens Experiment Report",
        "",
        f"- Suite: `{suite.name}`",
        f"- Description: {suite.description}",
        f"- Generated at: `{datetime.now(tz=UTC).isoformat()}`",
        f"- Parallel workers: `{suite.base_config.parallel_workers}`",
        "",
        "## Summary",
        "",
        _results_markdown_table(experiment_results),
        "",
    ]

    if any(record.status == "failed" for record in execution_records):
        lines.extend(["## Failures", ""])
        for record in execution_records:
            if record.status == "failed":
                lines.append(f"- `{record.method_name}` seed `{record.seed}` failed: `{record.error}`")
        lines.append("")

    overall = statistics.get("overall", {}) if isinstance(statistics, dict) else {}
    if overall:
        lines.extend(["## Robust Statistics", ""])
        lines.append(
            "- Friedman:"
            f" chi2={float(overall.get('statistic', 0.0)):+.3f},"
            f" p={float(overall.get('p_value', 1.0)):.4g},"
            f" methods={int(overall.get('n_methods', 0))},"
            f" blocks={int(overall.get('n_blocks', 0))}"
        )
        significant_pairs = [
            item
            for item in overall.get("posthoc", [])
            if isinstance(item, dict) and bool(item.get("significant"))
        ]
        if significant_pairs:
            for item in significant_pairs:
                lines.append(
                    f"- Posthoc significant: `{item['left']}` vs `{item['right']}`"
                    f" (rank gap {float(item['rank_gap']):+.3f}, CD {float(item['critical_difference']):+.3f})"
                )
        else:
            lines.append("- Posthoc significant pairs: none")
        lines.append("")

    baseline_name = statistics.get("baselineMethod") if isinstance(statistics, dict) else None
    comparisons = statistics.get("vsBaseline", {}) if isinstance(statistics, dict) else {}
    if baseline_name and isinstance(comparisons, dict) and comparisons:
        lines.extend([f"## Delta vs {baseline_name}", ""])
        for name, payload in comparisons.items():
            welch = payload.get("welch", {})
            robust = payload.get("robust", {})
            wilcoxon = robust.get("wilcoxon", {})
            bayesian = robust.get("bayesian", {})
            bootstrap_ci = robust.get("bootstrap_ci", [0.0, 0.0])
            lines.append(
                f"- `{name}`:"
                f" t={float(welch.get('t_statistic', 0.0)):+.3f},"
                f" p={float(welch.get('p_value', 1.0)):.4g},"
                f" d={float(welch.get('effect_size_cohens_d', 0.0)):+.3f};"
                f" Wilcoxon p={float(wilcoxon.get('p_value', 1.0)):.4g};"
                f" bootstrap CI=[{float(bootstrap_ci[0]):+.4f}, {float(bootstrap_ci[1]):+.4f}];"
                f" P(>{baseline_name})={float(bayesian.get('prob_left_gt_right', 0.5)):.3f}"
            )
        lines.append("")

    lines.extend(["## LaTeX", "", "```latex", _render_latex_report(experiment_results), "```", ""])
    return "\n".join(lines)


def _suite_report(
    suite: SuitePlan,
    experiment_results: dict[str, ExperimentResult],
    execution_records: list[ExecutionRecord],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics = _build_report_statistics(experiment_results)
    report = {
        "generatedAt": datetime.now(tz=UTC).isoformat(),
        "suite": suite.name,
        "description": suite.description,
        "baseConfig": _config_snapshot(suite.base_config),
        "specs": [spec.to_manifest() for spec in suite.specs],
        "statistics": statistics,
        "results": {method: _experiment_result_to_dict(result) for method, result in experiment_results.items()},
        "executionRecords": [_execution_record_to_dict(record) for record in execution_records],
    }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    (output_dir / "report.md").write_text(
        _render_markdown_report(suite, experiment_results, execution_records, statistics),
        encoding="utf-8",
    )
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


def _load_records_for_spec(output_root: Path, spec: ExperimentSpec) -> list[ExecutionRecord]:
    records_path = output_root / "execution_records.json"
    if not records_path.exists():
        return []
    data = json.loads(records_path.read_text(encoding="utf-8"))
    records: list[ExecutionRecord] = []
    for item in data:
        if item.get("specKey") != spec.key:
            continue
        records.append(_execution_record_from_dict(item))
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
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    if output_root is None:
        output_root = base_config.artifact_root / "_experiments" / suite.name
    output_root = output_root / timestamp

    jobs = _build_jobs(suite, output_root, dry_run=dry_run)
    job_payloads = _execute_jobs(jobs, suite.base_config.parallel_workers)
    experiment_results, execution_records = _seed_results_from_job_payloads(suite, job_payloads)

    outputs: dict[str, str] = {}
    if not dry_run:
        outputs = _suite_report(suite, experiment_results, execution_records, output_root)

    return {
        "suite": suite.name,
        "description": suite.description,
        "baseConfig": _config_snapshot(suite.base_config),
        "specs": [spec.to_manifest() for spec in suite.specs],
        "statistics": _build_report_statistics(experiment_results),
        "experimentResults": {name: _experiment_result_to_dict(result) for name, result in experiment_results.items()},
        "executionRecords": [_execution_record_to_dict(record) for record in execution_records],
        "outputs": outputs,
    }


def report_experiments(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite = SuitePlan(
        name=str(manifest.get("suite", "unknown")),
        specs=[_spec_from_manifest(item) for item in manifest.get("specs", [])],
        base_config=config_from_snapshot(manifest.get("baseConfig", {})),
        description=str(manifest.get("description", "")),
    )
    output_root = manifest_path.parent

    experiment_results: dict[str, ExperimentResult] = {}
    execution_records: list[ExecutionRecord] = []

    for spec in suite.specs:
        records = _load_records_for_spec(output_root, spec)
        execution_records.extend(records)
        seed_results = [
            _record_to_seed_result(record)
            for record in records
            if record.status == "completed" and record.result.get("seedResult")
        ]
        experiment_results[spec.method_name] = ExperimentResult(method_name=spec.method_name, seed_results=seed_results)

    outputs = _suite_report(suite, experiment_results, execution_records, output_root)
    return {
        "suite": suite.name,
        "manifestPath": str(manifest_path),
        "outputs": outputs,
        "statistics": _build_report_statistics(experiment_results),
        "experimentResults": {name: _experiment_result_to_dict(result) for name, result in experiment_results.items()},
    }


def _build_base_config(args: argparse.Namespace) -> TrainingConfig:
    config = (
        load_training_config(args.config)
        if getattr(args, "config", None) is not None
        else TrainingConfig(experiment_name=getattr(args, "experiment_name", None) or "regime_lens_experiments")
    )
    updates: dict[str, Any] = {}

    if getattr(args, "experiment_name", None) is not None:
        updates["experiment_name"] = args.experiment_name
    if getattr(args, "artifact_root", None) is not None:
        updates["artifact_root"] = Path(args.artifact_root)
    if getattr(args, "seeds", None) is not None:
        updates["seeds"] = _parse_int_tuple(args.seeds)
    if getattr(args, "eval_seeds", None) is not None:
        updates["fixed_eval_seeds"] = _parse_int_tuple(args.eval_seeds)
    if getattr(args, "episodes", None) is not None:
        updates["episodes"] = int(args.episodes)
    if getattr(args, "checkpoint_interval", None) is not None:
        updates["checkpoint_interval"] = int(args.checkpoint_interval)
    if getattr(args, "metrics_flush_interval", None) is not None:
        updates["metrics_flush_interval"] = int(args.metrics_flush_interval)
    if getattr(args, "evaluation_episodes", None) is not None:
        updates["evaluation_episodes"] = int(args.evaluation_episodes)
    if getattr(args, "device", None) is not None:
        updates["device"] = args.device
    if getattr(args, "cpu_threads", None) is not None:
        updates["cpu_threads"] = int(args.cpu_threads)
    if getattr(args, "process_priority", None) is not None:
        updates["process_priority"] = args.process_priority
    if getattr(args, "resume_run", None) is not None:
        updates["resume_run_id"] = args.resume_run
    if getattr(args, "resume_checkpoint", None) is not None:
        updates["resume_checkpoint_id"] = args.resume_checkpoint
    if getattr(args, "algorithm", None) is not None:
        updates["algorithm"] = AlgorithmType(args.algorithm)
    if getattr(args, "continuous_actions", False):
        updates["continuous_actions"] = True
    if getattr(args, "data_source", None) is not None:
        updates["data_source"] = DataSource(args.data_source)
    if getattr(args, "data_cache_path", None) is not None:
        updates["data_cache_path"] = Path(args.data_cache_path)
    if getattr(args, "real_data_symbols", None) is not None:
        updates["real_data_symbols"] = _parse_str_tuple(args.real_data_symbols)
    if getattr(args, "nonstationary_mode", None) is not None:
        updates["nonstationary_mode"] = str(args.nonstationary_mode)
    if getattr(args, "nonstationary_drift_scale", None) is not None:
        updates["nonstationary_drift_scale"] = float(args.nonstationary_drift_scale)
    if getattr(args, "gate_type", None) is not None:
        updates["gate_type"] = GateType(args.gate_type)
    if getattr(args, "context_len", None) is not None:
        updates["context_len"] = int(args.context_len)
    if getattr(args, "tracking_backend", None) is not None:
        updates["tracking_backend"] = TrackingBackend(args.tracking_backend)
    if getattr(args, "parallel_workers", None) is not None:
        updates["parallel_workers"] = max(1, int(args.parallel_workers))

    return replace(config, **updates)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(value) for value in values)


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _result_from_experiment_dict(payload: dict[str, Any]) -> ExperimentResult:
    return _result_from_dict(payload)


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
        output_root=base_config.artifact_root / "_experiments" / base_config.experiment_name,
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


def _cmd_serve(args: argparse.Namespace) -> int:
    base_config = _build_base_config(args)
    if __package__ in {None, ""}:
        from backend.regime_lens.web import run_server
    else:
        from .web import run_server

    run_server(base_config.artifact_root, host=args.host, port=args.port)
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
        f"Parallel workers: {suite.base_config.parallel_workers}",
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
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML/JSON config file to load first.")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Override the artifact root.")
    parser.add_argument("--experiment-name", default=None, help="Override the experiment label written to artifacts.")
    parser.add_argument("--seeds", default=None, help="Comma-separated training seeds. Defaults to the config seeds.")
    parser.add_argument("--eval-seeds", default=None, help="Comma-separated evaluation seeds. Defaults to the config eval seeds.")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes.")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Override checkpoint interval.")
    parser.add_argument("--metrics-flush-interval", type=int, default=None, help="Override metrics flush interval.")
    parser.add_argument("--evaluation-episodes", type=int, default=None, help="Override evaluation episodes per checkpoint.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU thread override.")
    parser.add_argument("--process-priority", default=None, help="Process priority hint.")
    parser.add_argument("--resume-run", default=None, help="Resume training from an existing run ID.")
    parser.add_argument("--resume-checkpoint", default=None, help="Resume training from a specific checkpoint ID.")
    parser.add_argument("--algorithm", default=None, choices=tuple(item.value for item in AlgorithmType), help="Override the learning algorithm family.")
    parser.add_argument("--continuous-actions", action="store_true", help="Enable the continuous-action environment path in config.")
    parser.add_argument("--data-source", default=None, choices=tuple(item.value for item in DataSource), help="Override the regime parameter data source.")
    parser.add_argument("--data-cache-path", default=None, help="Path to cached real-data CSV input.")
    parser.add_argument("--real-data-symbols", default=None, help="Comma-separated symbol list for real-data fitting or multi-asset configs.")
    parser.add_argument("--nonstationary-mode", default=None, help="Override the non-stationary regime drift mode.")
    parser.add_argument("--nonstationary-drift-scale", type=float, default=None, help="Override the non-stationary drift scale.")
    parser.add_argument("--gate-type", default=None, choices=tuple(item.value for item in GateType), help="Override the RCMoE gate type.")
    parser.add_argument("--context-len", type=int, default=None, help="Override temporal gate context length.")
    parser.add_argument("--tracking-backend", default=None, choices=tuple(item.value for item in TrackingBackend), help="Override experiment tracking backend.")
    parser.add_argument("--parallel-workers", type=int, default=None, help="Run seeds/specs in parallel using isolated artifact roots.")
    parser.add_argument("--ood-kind", default="all", choices=("all", "persistence", "switch", "volatility", "drift"), help="Which OOD subset to use.")
    parser.add_argument("--ablation-kind", default="all", choices=("all", "experts", "gate", "lb", "hidden"), help="Which ablation subset to use.")


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML/JSON config file to load first.")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Override the artifact root to browse.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the dashboard.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the dashboard.")


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
    run_parser.add_argument("--dry-run", action="store_true", help="Build the plan but skip execution.")
    run_parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output.")

    report_parser = subparsers.add_parser("report", help="Rebuild a report from an existing manifest.")
    report_parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json created by this runner.")
    report_parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output.")

    serve_parser = subparsers.add_parser("serve", help="Serve the artifact dashboard.")
    _add_serve_args(serve_parser)

    args = parser.parse_args(argv)
    command = args.command or "plan"

    if command == "plan":
        return _cmd_plan(args)
    if command == "run":
        return _cmd_run(args)
    if command == "report":
        return _cmd_report(args)
    if command == "serve":
        return _cmd_serve(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
