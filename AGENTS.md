# Agent Maintenance Guide

This repo is the RegimeForge project. The public name is `RegimeForge`; the Python package is
`regime_lens`.

## Local Commands

Use the known-good Windows interpreter unless the user provides a different environment:

```powershell
cd D:\RL\backend
D:\miniconda\envs\statshell\python.exe -m ruff check D:\RL\backend\regime_lens D:\RL\backend\tests
D:\miniconda\envs\statshell\python.exe -m pytest D:\RL\backend\tests -q
D:\miniconda\envs\statshell\python.exe -m regime_lens.run_experiments plan --suite smoke
git -C D:\RL diff --check
```

## Architecture Rules

- Keep `TrainingManager` focused on orchestration: run lifecycle, training loops, artifact writes, live telemetry, and checkpoint evaluation.
- Put environment-to-agent observation shaping in `backend/regime_lens/agent_io.py`.
- Do not duplicate Oracle one-hot, HMM posterior, temporal context, Transformer sequence, or neutral policy-surface logic inside `training.py`.
- When adding an agent, update `AgentType`, `_create_agent`, `AgentObservationAdapter`, tests, README, `backend/README.md`, and `docs/architecture.md`.
- Treat `backend/artifacts*`, `.ruff_cache`, `.pytest_cache`, `.venv`, and `.claude` as local runtime output, not source.

## Regression Tests To Add With Agent Changes

- A smoke test that the new agent can complete one checkpoint.
- A policy-surface or explainability shape test if the agent changes observation shape.
- A train/eval consistency test when adding Oracle, HMM, temporal, sequence, or continuous paths.

## Design Boundary

The fragile part of this project is not model code alone; it is keeping train, eval, policy surface,
explainability, and checkpoint resume on the same observation contract. Preserve that boundary before
expanding algorithm features.
