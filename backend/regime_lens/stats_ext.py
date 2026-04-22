from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return (0.0, 0.0)
    if len(array) == 1:
        only = float(array[0])
        return (only, only)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_resamples, len(array)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def wilcoxon_signed_rank(
    left: np.ndarray | list[float],
    right: np.ndarray | list[float],
) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return {"statistic": 0.0, "p_value": 1.0, "n": 0}
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "n": int(len(a))}
    statistic, p_value = stats.wilcoxon(a, b, zero_method="zsplit")
    return {"statistic": float(statistic), "p_value": float(p_value), "n": int(len(a))}


def friedman_with_posthoc(
    samples: dict[str, np.ndarray | list[float]],
) -> dict[str, Any]:
    aligned = {
        method: np.asarray(values, dtype=np.float64)
        for method, values in samples.items()
        if len(values) > 0
    }
    if len(aligned) < 3:
        return {"statistic": 0.0, "p_value": 1.0, "n_methods": len(aligned), "posthoc": []}
    lengths = {len(values) for values in aligned.values()}
    if len(lengths) != 1:
        return {"statistic": 0.0, "p_value": 1.0, "n_methods": len(aligned), "posthoc": []}
    statistic, p_value = stats.friedmanchisquare(*aligned.values())
    mean_ranks = _mean_ranks(aligned)
    methods = list(aligned)
    n_methods = len(methods)
    n_blocks = len(next(iter(aligned.values())))
    cd = float(stats.studentized_range.ppf(0.95, n_methods, np.inf) * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks)))
    posthoc: list[dict[str, Any]] = []
    for idx, left in enumerate(methods):
        for right in methods[idx + 1 :]:
            rank_gap = abs(mean_ranks[left] - mean_ranks[right])
            posthoc.append(
                {
                    "left": left,
                    "right": right,
                    "rank_gap": float(rank_gap),
                    "critical_difference": cd,
                    "significant": bool(rank_gap > cd),
                }
            )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_methods": n_methods,
        "n_blocks": n_blocks,
        "mean_ranks": mean_ranks,
        "critical_difference": cd,
        "posthoc": posthoc,
    }


def bayesian_effect_summary(
    left: np.ndarray | list[float],
    right: np.ndarray | list[float],
    *,
    n_resamples: int = 4_000,
    seed: int = 42,
) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return {"mean_diff": 0.0, "prob_left_gt_right": 0.5, "credible_interval": [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    left_samples = rng.choice(a, size=(n_resamples, len(a)), replace=True).mean(axis=1)
    right_samples = rng.choice(b, size=(n_resamples, len(b)), replace=True).mean(axis=1)
    deltas = left_samples - right_samples
    return {
        "mean_diff": float(np.mean(deltas)),
        "prob_left_gt_right": float(np.mean(deltas > 0.0)),
        "credible_interval": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def _mean_ranks(samples: dict[str, np.ndarray]) -> dict[str, float]:
    methods = list(samples)
    matrix = np.vstack([samples[method] for method in methods]).T
    ranks = np.vstack([stats.rankdata(-row, method="average") for row in matrix])
    means = ranks.mean(axis=0)
    return {method: float(means[idx]) for idx, method in enumerate(methods)}
