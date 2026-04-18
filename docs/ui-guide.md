# RegimeForge UI Guide

This guide explains the terminal dashboard in `backend/regime_lens/tui.py`.

## Overview

The UI is organized around five views:

1. `Overview`
2. `Regime Lens`
3. `Expert Deep Dive`
4. `Performance`
5. `Config`

The dashboard shows live training data when a run is active. If training has finished, the regime and expert panels fall back to checkpoint analysis so the UI still stays useful.

## Shortcuts

- `1` to `5`
  - Switch between the five views.

- `Tab`
  - Move focus to the next panel in the current view.

- `Shift+Tab`
  - Move focus to the previous panel in the current view.

- `Space`
  - Pause or resume training.

- `r`
  - Toggle extra regime details.

- `e`
  - Toggle extra expert details.

- `q` or `Esc`
  - Exit the dashboard.

## View-by-view

### Overview

This is the default control room. It combines the most important live signals:

- Current episode and total planned episodes
- Global step count
- Training speed
- Device and agent type
- Latest reward and strategy return
- Epsilon
- Gate accuracy
- Recent training spark lines
- Regime lens summary
- Performance summary
- Latest checkpoint summary
- Expert activation summary

Use this view when you want one screen that tells you whether training is healthy.

### Regime Lens

This view focuses on regime recognition and routing behavior.

It typically shows:

- Gate weights over the most recent steps
- The current or latest ground-truth regime
- Rolling gate accuracy
- Regime analysis metrics from checkpoint evaluation
- A recent regime timeline

Metrics you will see here include:

- `Gate Acc`
  - How often the gate matches the true regime.

- `NMI`
  - Mutual-information-based alignment between hard gate assignments and regimes.

- `ARI`
  - Cluster agreement between gate assignments and regimes.

- `Gate Entropy`
  - Routing uncertainty. Lower usually means more confident expert selection.

### Expert Deep Dive

This view focuses on expert specialization.

It typically shows:

- Current expert routing weights
- Expert utilization
- A regime-by-expert activation heatmap
- Specialization score
- Optional checkpoint-based expert analysis when live telemetry is unavailable

Key indicators:

- `Expert Utilization`
  - How often each expert is selected as the top route.

- `Specialization Score`
  - How strongly the experts separate by regime.

- `Activation Matrix`
  - Average gate weight per regime and expert.

### Performance

This view focuses on trading quality and risk.

It typically shows:

- Return and reward summaries
- Risk-adjusted metrics
- Baseline comparisons
- Per-regime performance breakdown

The most important indicators are:

- `Cumulative Return`
  - Total return over the evaluation window.

- `Sharpe`
  - Return adjusted by total volatility.

- `Sortino`
  - Return adjusted by downside volatility only.

- `Max Drawdown`
  - Largest peak-to-trough loss.

- `Calmar`
  - Return divided by drawdown.

- `Win Rate`
  - Fraction of steps with positive PnL.

- `Profit Factor`
  - Gross profit divided by gross loss.

### Config

This view shows the current experiment settings and runtime state:

- Agent type
- Seed settings
- Episode and checkpoint settings
- Device and process priority
- Regime transition matrix
- Regime parameters

Use this view when you want to verify that a run was launched with the expected configuration.

## Practical reading order

If you are watching training live, read the UI in this order:

1. `Overview` for health and progress.
2. `Regime Lens` for routing quality.
3. `Expert Deep Dive` for specialization.
4. `Performance` for trading quality.
5. `Config` for reproducibility.

## When panels stay empty

Some panels only become meaningful for `rcmoe_dqn`.

If you run plain `dqn`, there is no gate or expert system, so the regime and expert panels will have limited or no data. That is expected behavior, not a UI bug.

