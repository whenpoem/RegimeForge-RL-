# RegimeForge Backend

This directory contains the installable Python package that powers RegimeForge.
The package name is `regime_lens`.

## What Lives Here

- Synthetic market generation and hidden-regime dynamics
- DQN, Oracle DQN, HMM+DQN, and RCMoE-DQN implementations
- Training orchestration, artifact writing, and checkpoint evaluation
- Rich terminal dashboard
- Experiment runner and report generation utilities
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
D:\miniconda\envs\statshell\python.exe -m backend.regime_lens.run_experiments plan --suite full
```

Run a smoke suite:

```powershell
D:\miniconda\envs\statshell\python.exe -m backend.regime_lens.run_experiments run --suite smoke --experiment-name smoke_demo
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
            |-- regime_analysis.json
            `-- expert_analysis.json
```

Experiment bundles are written under `artifacts/_experiments/` and
include `report.json`, `report.md`, `results.tex`, `manifest.json`, and execution records.

## Packaging Notes

- Public project name: `RegimeForge`
- Python package name: `regime_lens`
- Build metadata lives in [`pyproject.toml`](pyproject.toml)

For higher-level project documentation, start with the repository [README](../README.md).
