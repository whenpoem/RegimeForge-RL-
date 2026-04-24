"""Multi-seed experiment runner and statistical testing framework.

Provides infrastructure for running the same RL method across multiple
random seeds, collecting per-seed metrics, computing confidence intervals,
and performing pairwise significance tests between methods.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Callable

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from .stats_ext import bayesian_effect_summary, bootstrap_ci, friedman_with_posthoc, wilcoxon_signed_rank


@dataclass
class SeedResult:
    """Metrics from a single seed run."""

    seed: int
    cumulative_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_eval_reward: float = 0.0
    per_regime: dict[str, dict[str, float]] = field(default_factory=dict)
    gate_nmi: float | None = None
    gate_ari: float | None = None
    expert_utilization: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Aggregated result across seeds for one method."""

    method_name: str
    seed_results: list[SeedResult] = field(default_factory=list)

    def _values(self, attr: str) -> list[float]:
        return [getattr(result, attr) for result in self.seed_results]

    @property
    def returns(self) -> list[float]:
        return self._values("cumulative_return")

    @property
    def sharpes(self) -> list[float]:
        return self._values("sharpe")

    @property
    def max_drawdowns(self) -> list[float]:
        return self._values("max_drawdown")

    def mean(self, attr: str = "cumulative_return") -> float:
        values = self._values(attr)
        return float(np.mean(values)) if values else 0.0

    def std(self, attr: str = "cumulative_return") -> float:
        values = self._values(attr)
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    def ci_95(self, attr: str = "cumulative_return") -> tuple[float, float]:
        """95% confidence interval using a t-distribution."""

        values = self._values(attr)
        n = len(values)
        if n == 0:
            return (0.0, 0.0)
        if n == 1:
            only = float(values[0])
            return (only, only)
        mean = float(np.mean(values))
        se = float(stats.sem(values))
        lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
        return (float(lo), float(hi))

    def bootstrap_ci(self, attr: str = "cumulative_return") -> tuple[float, float]:
        """95% bootstrap confidence interval."""

        return bootstrap_ci(self._values(attr))

    def vs_baseline(self, baseline: "ExperimentResult", attr: str = "cumulative_return") -> dict[str, Any]:
        """Welch's t-test + Cohen's d against a baseline method."""

        a = np.asarray(self._values(attr), dtype=np.float64)
        b = np.asarray(baseline._values(attr), dtype=np.float64)
        if len(a) < 2 or len(b) < 2:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant_005": False,
                "significant_001": False,
                "effect_size_cohens_d": 0.0,
            }
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_005": bool(p_value < 0.05),
            "significant_001": bool(p_value < 0.01),
            "effect_size_cohens_d": _cohens_d(a, b),
        }

    def robust_vs_baseline(self, baseline: "ExperimentResult", attr: str = "cumulative_return") -> dict[str, Any]:
        """Robust paired/non-parametric statistics against a baseline method."""

        a = np.asarray(self._values(attr), dtype=np.float64)
        b = np.asarray(baseline._values(attr), dtype=np.float64)
        if len(a) == 0 or len(b) == 0:
            return {
                "bootstrap_ci": [0.0, 0.0],
                "wilcoxon": {"statistic": 0.0, "p_value": 1.0, "n": 0},
                "bayesian": {"mean_diff": 0.0, "prob_left_gt_right": 0.5, "credible_interval": [0.0, 0.0]},
            }
        pair_count = min(len(a), len(b))
        paired_a = a[:pair_count]
        paired_b = b[:pair_count]
        return {
            "bootstrap_ci": list(bootstrap_ci(a)),
            "wilcoxon": wilcoxon_signed_rank(paired_a, paired_b),
            "bayesian": bayesian_effect_summary(a, b),
        }

    def summary_row(self, attr: str = "cumulative_return") -> str:
        """One-line summary: mean +/- std [CI_lo, CI_hi]."""

        lo, hi = self.ci_95(attr)
        return f"{self.mean(attr):+.4f} +/- {self.std(attr):.4f}  [{lo:+.4f}, {hi:+.4f}]"


def run_multi_seed(
    config: Any,
    seeds: list[int] | tuple[int, ...],
    method_name: str,
    method_factory: Callable[[Any], SeedResult | dict[str, Any]],
) -> ExperimentResult:
    """Execute the same method across multiple seeds and aggregate results."""

    result = ExperimentResult(method_name=method_name)
    for seed in seeds:
        seed_config = _config_with_seed(config, int(seed))
        payload = method_factory(seed_config)
        if isinstance(payload, SeedResult):
            seed_result = payload
        elif isinstance(payload, dict):
            seed_result = SeedResult(seed=int(seed), **payload)
        else:
            raise TypeError(f"Unsupported seed payload type: {type(payload)!r}")
        if seed_result.seed != int(seed):
            seed_result.seed = int(seed)
        result.seed_results.append(seed_result)
    return result


def compare_methods_robustly(
    results: dict[str, ExperimentResult],
    attr: str = "cumulative_return",
) -> dict[str, Any]:
    samples = {
        method: np.asarray(result._values(attr), dtype=np.float64)
        for method, result in results.items()
        if result.seed_results
    }
    return friedman_with_posthoc(samples)


def results_to_latex(
    results: dict[str, ExperimentResult],
    metrics: tuple[str, ...] = ("cumulative_return", "sharpe", "max_drawdown"),
    metric_labels: tuple[str, ...] = ("Return", "Sharpe", "MDD"),
    highlight_best: bool = True,
) -> str:
    """Generate a LaTeX table comparing methods across metrics."""

    methods = list(results.keys())
    header = " & ".join(["Method"] + list(metric_labels)) + r" \\"
    lines = [
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    best_vals: dict[str, float] = {}
    if highlight_best and methods:
        for metric in metrics:
            vals = {method: results[method].mean(metric) for method in methods}
            best_vals[metric] = max(vals.values())

    for method in methods:
        experiment = results[method]
        cells = [method]
        for metric in metrics:
            mean = experiment.mean(metric)
            std = experiment.std(metric)
            cell = f"{mean:+.4f} +/- {std:.4f}"
            if highlight_best and abs(mean - best_vals.get(metric, mean)) < 1e-10:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def results_to_latex_table(
    results: dict[str, ExperimentResult],
    metrics: tuple[str, ...] = ("cumulative_return", "sharpe", "max_drawdown"),
    metric_labels: tuple[str, ...] = ("Return", "Sharpe", "MDD"),
    highlight_best: bool = True,
) -> str:
    """Backward-compatible alias matching the implementation plan name."""

    return results_to_latex(
        results,
        metrics=metrics,
        metric_labels=metric_labels,
        highlight_best=highlight_best,
    )


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = float(np.var(a, ddof=1))
    var2 = float(np.var(b, ddof=1))
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / (pooled + 1e-10))


def _config_with_seed(config: Any, seed: int) -> Any:
    if is_dataclass(config):
        return replace(config, seed=seed)
    cloned = deepcopy(config)
    cloned.seed = seed
    return cloned
