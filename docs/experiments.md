# RegimeForge Experiments

This document covers the experiment runner in `backend/regime_lens/run_experiments.py`.

## Experiment suites

The runner supports these suites:

- `smoke`
  - Fast validation suite.
  - Short episodes, fewer evaluation seeds, and a small method set.

- `full`
  - Main benchmark matrix.
  - Compares the core agent variants and baselines.

- `ablation`
  - Sweeps architecture or regularization choices for RCMoE.

- `ood`
  - Out-of-distribution generalization tests.
  - Supports persistence, fast switching, high volatility, and drift variants.

- `all`
  - Combines the full benchmark, ablations, and OOD sweeps.

Common OOD modes:

- `persistence`
  - More persistent regimes.

- `switch`
  - Faster regime switching.

- `volatility`
  - Higher-volatility regime dynamics.

- `drift`
  - Non-stationary regime drift for continuous robustness checks.

Common ablation modes:

- `experts`
- `gate`
- `lb`
- `hidden`

## Baselines and methods

The main comparison set includes:

- `Vanilla DQN`
- `Oracle DQN`
- `RCMoE-DQN`
- `RCMoE-DQN+LB`
- `HMM+DQN`
- `Random`
- `Buy&Hold`

What they mean:

- `Vanilla DQN`
  - Standard baseline without regime awareness.

- `Oracle DQN`
  - Upper-bound baseline that receives the true regime as an additional input.

- `RCMoE-DQN`
  - Regime-aware mixture-of-experts agent.

- `RCMoE-DQN+LB`
  - Same as above, with load-balancing regularization.

- `HMM+DQN`
  - Detector-plus-policy baseline.

- `Random`
  - Random action policy.

- `Buy&Hold`
  - Always-long benchmark.

## Common commands

Plan a suite without running it:

```bash
python -m regime_lens.run_experiments plan --suite smoke
```

Run the smoke suite:

```bash
python -m regime_lens.run_experiments run --suite smoke --experiment-name regimeforge_smoke
```

Run the full benchmark with a custom device:

```bash
python -m regime_lens.run_experiments run --suite full --device cpu --experiment-name regimeforge_full
```

Run an ablation sweep:

```bash
python -m regime_lens.run_experiments run --suite ablation --ablation-kind experts --experiment-name regimeforge_ablation
```

Run an OOD sweep:

```bash
python -m regime_lens.run_experiments run --suite ood --ood-kind volatility --experiment-name regimeforge_ood
```

Run a continuous-action smoke suite:

```bash
python -m regime_lens.run_experiments run --suite smoke --algorithm sac --continuous-actions --episodes 1 --evaluation-episodes 1
```

Run with local YAML or TOML config:

```bash
python -m regime_lens.run_experiments run --suite smoke --config experiments/smoke.yaml
```

Serve the artifact dashboard:

```bash
python -m regime_lens.run_experiments serve --artifact-root D:\RL\backend\artifacts
```

Rebuild a report from an existing manifest:

```bash
python -m regime_lens.run_experiments report --manifest D:\RL\backend\artifacts\_experiments\regimeforge_smoke\<timestamp>\manifest.json
```

Useful flags:

- `--artifact-root`
  - Override the artifact root directory.

- `--seeds`
  - Override training seeds.

- `--eval-seeds`
  - Override evaluation seeds.

- `--episodes`
  - Override training length.

- `--checkpoint-interval`
  - Override checkpoint cadence.

- `--metrics-flush-interval`
  - Override metrics flush cadence.

- `--evaluation-episodes`
  - Override evaluation episodes per checkpoint.

- `--dry-run`
  - Build the plan but skip execution.

- `--json`
  - Emit machine-readable output.

- `--config`
  - Load a YAML, TOML, or JSON config file with optional inheritance.

- `--algorithm`
  - Select `dqn`, `ppo`, or `sac`.

- `--continuous-actions`
  - Run the continuous environment path.

- `--parallel-workers`
  - Run specs/seeds in isolated worker artifact roots.

- `--resume-run` and `--resume-checkpoint`
  - Continue training from saved model and replay/rollout state.

## Result layout

Every suite run writes a timestamped bundle under:

`backend/artifacts/_experiments/<suite>/<timestamp>/`

Typical outputs:

- `manifest.json`
  - Suite definition and run metadata.

- `execution_records.json`
  - Per-seed execution status.

- `report.json`
  - Structured summary of the suite.

- `report.md`
  - Human-readable Markdown report.

- `results.tex`
  - LaTeX table for paper or appendix use.

Trained checkpoint directories also contain model weights, stats, resume state, explainability,
and reproducibility payloads. Reports are generated from those artifacts, so `report` can rebuild
the Markdown/JSON/LaTeX bundle from a saved `manifest.json`.

## How to read the results

When comparing methods, keep these rules in mind:

1. Compare methods on the same suite and the same seed set.
2. Use `mean` and confidence intervals, not only the best run.
3. Treat `smoke` as a pipeline check, not as a scientific result.
4. Look at both return and risk-adjusted metrics.
5. For RCMoE, inspect regime alignment and expert specialization, not just profit.

Interpretation shortcuts:

- High return with poor Sharpe usually means the strategy is noisy or unstable.
- Better Sharpe with lower drawdown is often preferable to slightly higher raw return.
- Strong `NMI` or `ARI` means the gate is learning regime structure, but it does not automatically guarantee better trading performance.
- A balanced expert utilization pattern is usually healthier than a collapsed single-expert pattern.

## Recommended workflow

1. Run `smoke` first to verify the pipeline.
2. Run `full` for headline comparisons.
3. Run `ablation` to test architectural choices.
4. Run `ood` to check generalization.
5. Export the report bundle and include the Markdown and LaTeX outputs in your GitHub release or paper notes.
