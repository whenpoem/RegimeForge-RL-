"""Financial evaluation metrics for RL trading strategies.

Provides standard quantitative-finance performance measures used to
evaluate and compare agent strategies across regimes, seeds, and
baselines.  All functions accept plain NumPy arrays so they can be
used both during training (streaming) and in post-hoc analysis.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core risk-adjusted return metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio (daily returns assumed)."""
    excess = np.asarray(returns, dtype=np.float64) - risk_free
    std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 1e-10
    return float(np.mean(excess) / (std + 1e-10) * np.sqrt(252))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (penalises downside volatility only)."""
    excess = np.asarray(returns, dtype=np.float64) - risk_free
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    return float(np.mean(excess) / (downside + 1e-10) * np.sqrt(252))


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (returned as a negative fraction)."""
    cumulative = np.asarray(cumulative_returns, dtype=np.float64)
    if len(cumulative) == 0:
        return 0.0
    # Interpret as equity curve (1 + cumulative PnL)
    equity = 1.0 + cumulative
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-10)
    return float(np.min(drawdown))


def calmar_ratio(returns: np.ndarray, cumulative_returns: np.ndarray) -> float:
    """Annualised Calmar ratio (annual return / |max drawdown|)."""
    mdd = abs(max_drawdown(cumulative_returns))
    annual_return = float(np.mean(returns)) * 252
    return float(annual_return / (mdd + 1e-10))


# ---------------------------------------------------------------------------
# Trade-level statistics
# ---------------------------------------------------------------------------

def win_rate(step_pnls: np.ndarray) -> float:
    """Fraction of steps with positive PnL."""
    pnls = np.asarray(step_pnls, dtype=np.float64)
    if len(pnls) == 0:
        return 0.0
    return float(np.mean(pnls > 0.0))


def profit_factor(step_pnls: np.ndarray) -> float:
    """Gross profit / gross loss.  > 1 means net profitable."""
    pnls = np.asarray(step_pnls, dtype=np.float64)
    gross_profit = float(np.sum(pnls[pnls > 0.0]))
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0.0])))
    return float(gross_profit / (gross_loss + 1e-10))


# ---------------------------------------------------------------------------
# Per-regime breakdown
# ---------------------------------------------------------------------------

def per_regime_metrics(
    step_returns: np.ndarray,
    regimes: list[str],
    regime_labels: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by market regime.

    Returns a dict keyed by regime label, each containing:
        mean_return, sharpe, volatility, count, win_rate
    """
    step_returns = np.asarray(step_returns, dtype=np.float64)
    result: dict[str, dict[str, float]] = {}
    for label in regime_labels:
        mask = np.array([r == label for r in regimes], dtype=bool)
        n = int(mask.sum())
        if n == 0:
            continue
        regime_ret = step_returns[mask]
        std = float(np.std(regime_ret, ddof=1)) if n > 1 else 0.0
        result[label] = {
            "mean_return": float(np.mean(regime_ret)),
            "sharpe": float(np.mean(regime_ret) / (std + 1e-10) * np.sqrt(252)),
            "volatility": std,
            "count": n,
            "win_rate": float(np.mean(regime_ret > 0.0)),
        }
    return result


# ---------------------------------------------------------------------------
# Episode-level aggregation helper
# ---------------------------------------------------------------------------

def episode_metrics(
    step_pnls: np.ndarray,
    step_regimes: list[str],
    regime_labels: tuple[str, ...],
) -> dict[str, float | dict]:
    """Compute a full metrics bundle for a single evaluation episode.

    Returns a flat dict ready for JSON serialisation.
    """
    pnls = np.asarray(step_pnls, dtype=np.float64)
    cumulative = np.cumsum(pnls)
    return {
        "cumulative_return": float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
        "sharpe": sharpe_ratio(pnls),
        "sortino": sortino_ratio(pnls),
        "max_drawdown": max_drawdown(cumulative),
        "calmar": calmar_ratio(pnls, cumulative),
        "win_rate": win_rate(pnls),
        "profit_factor": profit_factor(pnls),
        "per_regime": per_regime_metrics(pnls, step_regimes, regime_labels),
    }
