# RegimeForge Backend

This directory contains the installable Python package that powers RegimeForge.
The package name is `regime_lens`.

## What Lives Here

- Synthetic and continuous market generation with hidden-regime dynamics
- DQN, Oracle DQN, HMM+DQN, RCMoE-DQN, Transformer-DQN, world-model, PPO, SAC, and continuous RCMoE implementations
- Training orchestration, model checkpoints, resume state, artifact writing, and checkpoint evaluation
- Agent observation adapters that keep train/eval/surface/explainability input shapes consistent
- Rich terminal dashboard and FastAPI artifact dashboard
- Experiment runner, parallel execution, robust statistics, and report generation utilities
- Visualization and LaTeX export helpers

## Install

```powershell
cd D:\RL\backend
D:\miniconda\envs\statshell\python.exe -m pip install -e .
```

## Main Entry Points

Launch a fresh TUI run:

```powershell
D:\miniconda\envs\statshell\python.exe -m regime_lens.tui --fresh --lang en --charset unicode
```

Plan the benchmark suite:

```powershell
D:\miniconda\envs\statshell\python.exe -m regime_lens.run_experiments plan --suite full
```

Run a smoke suite:

```powershell
D:\miniconda\envs\statshell\python.exe -m regime_lens.run_experiments run --suite smoke --experiment-name smoke_demo
```

Run a continuous-action SAC smoke suite:

```powershell
D:\miniconda\envs\statshell\python.exe -m regime_lens.run_experiments run --suite smoke --algorithm sac --continuous-actions --episodes 1 --evaluation-episodes 1
```

Serve the artifact dashboard:

```powershell
D:\miniconda\envs\statshell\python.exe -m regime_lens.web --artifact-root D:\RL\backend\artifacts
```

## Artifact Conventions

Training runs are written under `artifacts/`.

Typical run structure:

```text
artifacts/
`-- run-<timestamp>/
    |-- summary.json
    |-- metrics.json
    `-- checkpoints/
        `-- ckpt-<episode>/
            |-- summary.json
            |-- stats.json
            |-- resume_state.json
            |-- repro.json
            |-- policy.json
            |-- embedding.json
            |-- explainability.json
            |-- regime_analysis.json
            |-- expert_analysis.json
            `-- weights/
```

Experiment bundles are written under `artifacts/_experiments/` and
include `report.json`, `report.md`, `results.tex`, `manifest.json`, and execution records.

## Agent Integration Contract

`regime_lens.training` owns orchestration, not agent-specific observation wiring. New agent families
should route their train/eval/policy-surface/explainability inputs through
`regime_lens.agent_io.AgentObservationAdapter`.

Keep these cases centralized there:

- true-regime one-hot inputs for Oracle agents
- HMM/GMM posterior augmentation
- temporal RCMoE flattened context windows
- Transformer flattened observation sequences
- neutral surface/explainability observations used outside live rollouts

## Packaging Notes

- Public project name: `RegimeForge`
- Python package name: `regime_lens`
- Build metadata lives in [`pyproject.toml`](pyproject.toml)

For higher-level project documentation, start with the repository [README](../README.md).
