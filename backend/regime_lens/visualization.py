from __future__ import annotations

"""Academic visualisation helpers for Regime Lens / RCMoE-DQN."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .config import ACTION_LABELS, REGIME_LABELS
from .experiment import ExperimentResult, SeedResult

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    BoundaryNorm = ListedColormap = Patch = None  # type: ignore[assignment]
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None

try:  # pragma: no cover
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    PCA = TSNE = StandardScaler = None  # type: ignore[assignment]

try:  # pragma: no cover
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore[assignment]

REGIME_COLORS: dict[str, str] = {
    "bull": "#10b981",
    "bear": "#ef4444",
    "chop": "#f59e0b",
    "shock": "#ec4899",
}

EXPERT_COLORS: tuple[str, ...] = (
    "#38bdf8",
    "#fb7185",
    "#a78bfa",
    "#f59e0b",
    "#34d399",
    "#f472b6",
    "#22d3ee",
    "#c084fc",
)

ACTION_COLORS: dict[str, str] = {
    "short": "#f87171",
    "flat": "#94a3b8",
    "long": "#4ade80",
}

DEFAULT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#1f2937",
    "axes.labelcolor": "#111827",
    "axes.titlecolor": "#111827",
    "text.color": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "grid.color": "#9ca3af",
    "grid.alpha": 0.16,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}

SERIES_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "strategyReturn": ("strategyReturn", "strategy_return", "cumulative_return", "totalReward"),
    "cumulative_return": ("cumulative_return", "strategyReturn", "totalReward"),
    "totalReward": ("totalReward", "strategyReturn", "cumulative_return"),
    "marketReturn": ("marketReturn", "market_return"),
    "sharpe": ("sharpe",),
    "sortino": ("sortino",),
    "maxDrawdown": ("maxDrawdown", "max_drawdown"),
    "max_drawdown": ("max_drawdown", "maxDrawdown"),
    "winRate": ("winRate", "win_rate"),
    "win_rate": ("win_rate", "winRate"),
    "avgLoss": ("avgLoss", "avg_loss"),
    "avg_loss": ("avg_loss", "avgLoss"),
    "gateAccuracy": ("gateAccuracy", "gate_accuracy"),
    "gate_accuracy": ("gate_accuracy", "gateAccuracy"),
}

__all__ = [
    "ACTION_COLORS",
    "DEFAULT_STYLE",
    "EXPERT_COLORS",
    "REGIME_COLORS",
    "export_latex_table",
    "plot_expert_heatmap",
    "plot_gate_evolution",
    "plot_policy_surface_comparison",
    "plot_regime_tsne",
    "plot_training_curves",
]


def plot_training_curves(
    results_dict: Mapping[str, Any],
    metric: str = "strategyReturn",
    smoothing_window: int = 1,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 180,
):
    _require_matplotlib()
    _apply_default_style()

    if not results_dict:
        raise ValueError("results_dict is empty; no curves can be plotted.")

    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=dpi)
    metric_aliases = _metric_aliases(metric)
    palette = plt.get_cmap("tab10")

    for idx, (method_name, value) in enumerate(results_dict.items()):
        run_series = _collect_training_series(value, metric_aliases)
        if not run_series:
            warnings.warn(f"No training series found for '{method_name}'. Skipping.", RuntimeWarning, stacklevel=2)
            continue

        aligned = _align_series(run_series, smoothing_window=smoothing_window)
        episodes = aligned["episodes"]
        means = aligned["mean"]
        lower = aligned["lower"]
        upper = aligned["upper"]
        color = palette(idx % palette.N)

        ax.plot(episodes, means, label=method_name, color=color, linewidth=2.2)
        ax.fill_between(episodes, lower, upper, color=color, alpha=0.16, linewidth=0)
        if len(episodes) > 0:
            ax.scatter([episodes[-1]], [means[-1]], color=color, s=24, zorder=3)

    ax.set_xlabel("Episode")
    ax.set_ylabel(_pretty_metric_name(metric))
    ax.set_title(title or f"Training Curves: {_pretty_metric_name(metric)}")
    ax.legend(loc="best")
    ax.margins(x=0.01)

    if save_path is not None:
        _save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()
    return fig


def plot_expert_heatmap(
    activation_matrix: Any,
    regime_labels: Sequence[str] = REGIME_LABELS,
    expert_labels: Sequence[str] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    annotate: bool = True,
    dpi: int = 180,
):
    _require_matplotlib()
    _apply_default_style()

    matrix = _extract_activation_matrix(activation_matrix)
    if matrix.ndim != 2:
        raise ValueError("activation_matrix must be two-dimensional.")

    n_regimes, n_experts = matrix.shape
    labels_y = list(regime_labels)[:n_regimes]
    labels_x = list(expert_labels) if expert_labels is not None else [f"E{i + 1}" for i in range(n_experts)]
    if len(labels_x) < n_experts:
        labels_x.extend(f"E{i + 1}" for i in range(len(labels_x), n_experts))

    fig, ax = plt.subplots(figsize=(1.6 * max(4, n_experts), 0.9 * max(4, n_regimes)), dpi=dpi)
    image = ax.imshow(matrix, cmap="magma", aspect="auto", origin="upper")
    ax.set_xticks(np.arange(n_experts), labels=labels_x)
    ax.set_yticks(np.arange(n_regimes), labels=labels_y)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Regime")
    ax.set_title(title or "Expert-Regime Activation Heatmap")

    if annotate:
        threshold = float(np.nanmax(matrix)) * 0.55 if np.isfinite(matrix).any() else 0.0
        for row in range(n_regimes):
            for col in range(n_experts):
                value = matrix[row, col]
                text_color = "white" if value >= threshold else "#111827"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color=text_color)

    cbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("Mean gate weight")

    if save_path is not None:
        _save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()
    return fig


def plot_regime_tsne(
    hidden_states: Any,
    regimes: Sequence[str] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 180,
):
    _require_matplotlib()
    _apply_default_style()

    embedding, labels = _extract_embedding_bundle(hidden_states, regimes)
    if embedding.size == 0:
        raise ValueError("hidden_states is empty; nothing to plot.")

    projected, used_method = _project_to_2d(embedding)
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=dpi)
    color_values = _regime_color_values(labels)
    cmap = _regime_cmap()
    norm = BoundaryNorm(np.arange(-0.5, len(REGIME_LABELS) + 1.5, 1.0), len(REGIME_LABELS) + 1)
    ax.scatter(projected[:, 0], projected[:, 1], c=color_values, cmap=cmap, norm=norm, s=16, alpha=0.82, linewidths=0)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title or f"Regime Embedding ({used_method})")
    _add_regime_legend(ax)

    if save_path is not None:
        _save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()
    return fig


def plot_policy_surface_comparison(
    policies_dict: Mapping[str, Any],
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 180,
):
    _require_matplotlib()
    _apply_default_style()

    if not policies_dict:
        raise ValueError("policies_dict is empty; no policy surfaces can be plotted.")

    methods = list(policies_dict.keys())
    n_methods = len(methods)
    n_cols = min(3, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.5 * n_rows),
        dpi=dpi,
        squeeze=False,
        constrained_layout=True,
    )

    cmap = ListedColormap([ACTION_COLORS.get(label, color) for label, color in zip(ACTION_LABELS, ("#f87171", "#94a3b8", "#4ade80"), strict=False)])
    norm = BoundaryNorm(np.arange(-0.5, len(ACTION_LABELS) + 0.5, 1.0), cmap.N)

    for index, method_name in enumerate(methods):
        row = index // n_cols
        col = index % n_cols
        ax = axes[row][col]
        payload = _load_json_like(policies_dict[method_name])
        surface = _policy_payload_to_grid(payload)
        x_values = surface["x_values"]
        y_values = surface["y_values"]
        grid = surface["grid"]

        ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(x_values)), [f"{x:.2f}" for x in x_values], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(y_values)), [f"{y:.2f}" for y in y_values])
        ax.set_xlabel("Trend gap pct")
        ax.set_ylabel("Volatility pct")

        checkpoint_id = payload.get("checkpointId")
        subtitle = f"{method_name}"
        if checkpoint_id:
            subtitle += f" | {checkpoint_id}"
        ax.set_title(subtitle)

        if grid.size <= 121:
            for y_idx in range(grid.shape[0]):
                for x_idx in range(grid.shape[1]):
                    value = grid[y_idx, x_idx]
                    if np.isnan(value):
                        continue
                    label = ACTION_LABELS[int(value)]
                    ax.text(x_idx, y_idx, label[0].upper(), ha="center", va="center", fontsize=9, color="white")

    for index in range(n_methods, n_rows * n_cols):
        row = index // n_cols
        col = index % n_cols
        axes[row][col].axis("off")

    handles = [Patch(facecolor=ACTION_COLORS.get(label, "#9ca3af"), label=label) for label in ACTION_LABELS]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle(title or "Policy Surface Comparison", y=1.02, fontsize=13)

    if save_path is not None:
        _save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()
    return fig


def plot_gate_evolution(
    gate_history: Any,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 180,
):
    _require_matplotlib()
    _apply_default_style()

    weights, steps, regimes = _extract_gate_history(gate_history)
    if weights.size == 0:
        raise ValueError("gate_history is empty; nothing to plot.")

    n_steps, n_experts = weights.shape
    if steps is None:
        steps = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=dpi)
    _shade_regime_segments(ax, steps, regimes)

    colors = [EXPERT_COLORS[i % len(EXPERT_COLORS)] for i in range(n_experts)]
    ax.stackplot(steps, weights.T, labels=[f"Expert {i + 1}" for i in range(n_experts)], colors=colors, alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gate weight")
    ax.set_title(title or "Gate Weight Evolution")
    ax.legend(loc="upper right", ncol=min(4, n_experts))

    if regimes is not None and len(regimes) == len(steps):
        _annotate_regime_trace(ax, steps, regimes)

    if save_path is not None:
        _save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()
    return fig


def export_latex_table(
    results: Mapping[str, Any],
    metrics: Sequence[str] = ("cumulative_return", "sharpe", "max_drawdown"),
    metric_labels: Sequence[str] | None = None,
    highlight_best: bool = True,
    include_ci: bool = False,
) -> str:
    if not results:
        raise ValueError("results is empty; no LaTeX table can be generated.")

    normalized = {name: _coerce_experiment_result(name, value) for name, value in results.items()}
    labels = list(metric_labels) if metric_labels is not None else [_pretty_metric_name(metric) for metric in metrics]
    if len(labels) != len(metrics):
        raise ValueError("metric_labels must have the same length as metrics.")

    lines = [
        r"\begin{tabular}{l" + "c" * len(metrics) + r"}",
        r"\toprule",
        "Method & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]

    best_values: dict[str, float] = {}
    if highlight_best:
        for metric in metrics:
            values = [result.mean(metric) for result in normalized.values()]
            if values:
                best_values[metric] = max(values)

    for method_name, result in normalized.items():
        cells = [_latex_escape(method_name)]
        for metric in metrics:
            mean = result.mean(metric)
            std = result.std(metric)
            cell = f"${mean:+.4f} \\pm {std:.4f}$"
            if include_ci:
                lo, hi = result.ci_95(metric)
                cell = f"${mean:+.4f} \\pm {std:.4f}$ [{lo:+.4f}, {hi:+.4f}]"
            if highlight_best and np.isclose(mean, best_values.get(metric, mean)):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for regime_lens.visualization"
            + (f": {_MATPLOTLIB_ERROR}" if _MATPLOTLIB_ERROR is not None else "")
        )


def _apply_default_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(DEFAULT_STYLE)


def _save_figure(fig: Any, save_path: str | Path, dpi: int) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _metric_aliases(metric: str) -> tuple[str, ...]:
    return SERIES_METRIC_ALIASES.get(metric, (metric,))


def _pretty_metric_name(metric: str) -> str:
    mapping = {
        "strategyReturn": "Strategy Return",
        "cumulative_return": "Cumulative Return",
        "totalReward": "Total Reward",
        "marketReturn": "Market Return",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "maxDrawdown": "Max Drawdown",
        "max_drawdown": "Max Drawdown",
        "winRate": "Win Rate",
        "win_rate": "Win Rate",
        "avgLoss": "Avg Loss",
        "avg_loss": "Avg Loss",
        "gateAccuracy": "Gate Accuracy",
        "gate_accuracy": "Gate Accuracy",
    }
    return mapping.get(metric, metric.replace("_", " ").replace("Return", " Return").title())


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _load_json_like(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Path):
        return json.loads(value.read_text(encoding="utf-8"))
    if isinstance(value, str):
        path = Path(value)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _collect_training_series(value: Any, metric_aliases: tuple[str, ...]) -> list[list[tuple[int, float]]]:
    payloads: list[Any] = []

    if isinstance(value, Mapping):
        if "series" in value:
            payloads.append(value["series"])
        elif "metrics" in value and isinstance(value["metrics"], Mapping) and "series" in value["metrics"]:
            payloads.append(value["metrics"]["series"])
        elif "runs" in value:
            payloads.extend(value["runs"])
        else:
            payloads.append(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if value and all(isinstance(item, Mapping) and _looks_like_series_item(item, metric_aliases) for item in value):
            payloads.append(value)
        else:
            payloads.extend(value)
    elif isinstance(value, (str, Path)):
        payloads.append(_load_json_like(value))
    else:
        return []

    series_list: list[list[tuple[int, float]]] = []
    for payload in payloads:
        if isinstance(payload, (str, Path)):
            payload = _load_json_like(payload)
        if isinstance(payload, Mapping) and "series" in payload:
            payload = payload["series"]
        if isinstance(payload, Mapping) and "metrics" in payload and isinstance(payload["metrics"], Mapping):
            payload = payload["metrics"].get("series", payload)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            series = _series_from_payload(payload, metric_aliases)
            if series:
                series_list.append(series)
    return series_list


def _looks_like_series_item(item: Mapping[str, Any], metric_aliases: tuple[str, ...]) -> bool:
    if "episode" in item:
        return True
    return any(alias in item for alias in metric_aliases)


def _series_from_payload(payload: Sequence[Any], metric_aliases: tuple[str, ...]) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            continue
        episode = int(entry.get("episode", entry.get("episodeIndex", index + 1)))
        value = None
        for alias in metric_aliases:
            if alias in entry and entry[alias] is not None:
                value = entry[alias]
                break
        if value is None:
            continue
        try:
            series.append((episode, float(value)))
        except (TypeError, ValueError):
            continue
    return series


def _smooth_series(series: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1 or len(series) < 2:
        return series
    episodes = np.asarray([episode for episode, _ in series], dtype=np.int64)
    values = np.asarray([value for _, value in series], dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values, kernel, mode="same")
    return list(zip(episodes.tolist(), smoothed.tolist(), strict=False))  # type: ignore[list-item]


def _align_series(series_list: list[list[tuple[int, float]]], smoothing_window: int = 1) -> dict[str, np.ndarray]:
    episode_map: dict[int, list[float]] = {}
    for series in series_list:
        if smoothing_window > 1:
            series = _smooth_series(series, smoothing_window)
        for episode, value in series:
            episode_map.setdefault(int(episode), []).append(float(value))

    if not episode_map:
        empty = np.asarray([], dtype=np.float64)
        return {"episodes": empty, "mean": empty, "lower": empty, "upper": empty}

    episodes = np.asarray(sorted(episode_map), dtype=np.int64)
    values = [np.asarray(episode_map[episode], dtype=np.float64) for episode in episodes]
    means = np.asarray([float(np.mean(v)) for v in values], dtype=np.float64)
    lower = np.empty_like(means)
    upper = np.empty_like(means)
    for idx, v in enumerate(values):
        lo, hi = _mean_ci(v)
        lower[idx] = lo
        upper[idx] = hi
    return {"episodes": episodes, "mean": means, "lower": lower, "upper": upper}


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    mean = float(np.mean(values))
    if len(values) == 1:
        return (mean, mean)
    if stats is not None:
        sem = float(stats.sem(values, ddof=1))
        if np.isfinite(sem) and sem > 0.0:
            lo, hi = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
            return (float(lo), float(hi))
    std = float(np.std(values, ddof=1))
    margin = 1.96 * std / np.sqrt(len(values))
    return (mean - margin, mean + margin)


def _extract_activation_matrix(activation_matrix: Any) -> np.ndarray:
    if isinstance(activation_matrix, np.ndarray):
        return activation_matrix.astype(np.float64, copy=False)
    if isinstance(activation_matrix, Mapping):
        if "activation_matrix" in activation_matrix:
            return np.asarray(activation_matrix["activation_matrix"], dtype=np.float64)
        if "regime_analysis" in activation_matrix and isinstance(activation_matrix["regime_analysis"], Mapping):
            return np.asarray(activation_matrix["regime_analysis"].get("activation_matrix", []), dtype=np.float64)
    return np.asarray(activation_matrix, dtype=np.float64)


def _extract_embedding_bundle(hidden_states: Any, regimes: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
    if isinstance(hidden_states, Mapping):
        if "points" in hidden_states:
            points = hidden_states["points"]
            if points and isinstance(points[0], Mapping) and {"x", "y"}.issubset(points[0].keys()):
                coords = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float64)
                labels = [str(p.get("regime", "")) for p in points]
                return coords, labels
        if "embeddingStates" in hidden_states:
            embedding = np.asarray(hidden_states["embeddingStates"], dtype=np.float64)
            labels = list(regimes or hidden_states.get("stepRegimes", []))
            return embedding, labels
        if "hidden_states" in hidden_states:
            embedding = np.asarray(hidden_states["hidden_states"], dtype=np.float64)
            labels = list(regimes or hidden_states.get("regimes", []))
            return embedding, labels

    embedding = np.asarray(hidden_states, dtype=np.float64)
    labels = list(regimes or [])
    if embedding.ndim == 1:
        embedding = embedding.reshape(-1, 1)
    if labels and len(labels) != len(embedding):
        labels = labels[: len(embedding)]
    if not labels:
        labels = ["unknown"] * len(embedding)
    return embedding, labels


def _project_to_2d(embedding: np.ndarray) -> tuple[np.ndarray, str]:
    if embedding.ndim != 2:
        raise ValueError("hidden_states must be a 2D array after normalization.")
    if embedding.shape[0] < 2:
        return np.zeros((embedding.shape[0], 2), dtype=np.float64), "degenerate"
    if embedding.shape[1] == 2:
        return embedding.astype(np.float64, copy=False), "2D"

    scaled = embedding
    if StandardScaler is not None:
        scaled = StandardScaler().fit_transform(embedding)

    if TSNE is not None and embedding.shape[0] >= 4:
        perplexity = min(30.0, max(2.0, float(embedding.shape[0] - 1) / 3.0))
        perplexity = min(perplexity, float(embedding.shape[0] - 1))
        try:
            projected = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=42,
            ).fit_transform(scaled)
            return projected.astype(np.float64, copy=False), "t-SNE"
        except Exception:
            pass

    if PCA is not None:
        try:
            projected = PCA(n_components=2, random_state=42).fit_transform(scaled)
            return projected.astype(np.float64, copy=False), "PCA"
        except Exception:
            pass

    projected = np.zeros((embedding.shape[0], 2), dtype=np.float64)
    if embedding.shape[1] > 0:
        projected[:, 0] = embedding[:, 0]
    if embedding.shape[1] > 1:
        projected[:, 1] = embedding[:, 1]
    return projected, "raw"


def _regime_cmap():
    if ListedColormap is None:
        return None
    return ListedColormap([REGIME_COLORS.get(label, "#64748b") for label in REGIME_LABELS] + ["#64748b"])


def _regime_color_values(labels: Sequence[str]) -> np.ndarray:
    regime_to_index = {label: idx for idx, label in enumerate(REGIME_LABELS)}
    return np.asarray([regime_to_index.get(str(label), len(REGIME_LABELS)) for label in labels], dtype=np.int64)


def _add_regime_legend(ax: Any) -> None:
    handles = [Patch(facecolor=REGIME_COLORS[label], edgecolor="none", label=label.title()) for label in REGIME_LABELS]
    ax.legend(handles=handles, title="Regime", loc="best")


def _extract_gate_history(gate_history: Any) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    if isinstance(gate_history, np.ndarray):
        weights = gate_history.astype(np.float64, copy=False)
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        return weights, None, None

    if isinstance(gate_history, Mapping):
        if "history" in gate_history:
            gate_history = gate_history["history"]
        elif "gateWeights" in gate_history or "weights" in gate_history:
            raw_weights = gate_history.get("gateWeights", gate_history.get("weights"))
            weights = np.asarray(raw_weights, dtype=np.float64)
            if weights.ndim == 1:
                weights = weights.reshape(-1, 1)
            raw_steps = gate_history.get("steps")
            raw_regimes = gate_history.get("regimes", gate_history.get("stepRegimes", gate_history.get("regimeLabels")))
            steps = None
            if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, (str, bytes, bytearray)):
                step_array = np.asarray(raw_steps, dtype=np.int64)
                if len(step_array) == len(weights):
                    steps = step_array
            regimes = None
            if isinstance(raw_regimes, Sequence) and not isinstance(raw_regimes, (str, bytes, bytearray)):
                regime_list = [str(item) for item in raw_regimes]
                if len(regime_list) == len(weights):
                    regimes = regime_list
            return weights, steps, regimes

    if not isinstance(gate_history, Sequence) or isinstance(gate_history, (str, bytes, bytearray)):
        return np.asarray([]), None, None

    weights: list[list[float]] = []
    steps: list[int] = []
    regimes: list[str] = []
    have_steps = False
    have_regimes = False

    for index, item in enumerate(gate_history):
        if isinstance(item, Mapping):
            row = item.get("weights", item.get("gateWeights", item.get("gate_weights")))
            if row is None:
                continue
            weights.append([float(value) for value in row])
            if "step" in item:
                have_steps = True
                steps.append(int(item["step"]))
            if "regime" in item:
                have_regimes = True
                regimes.append(str(item["regime"]))
            elif "regimeLabel" in item:
                have_regimes = True
                regimes.append(str(item["regimeLabel"]))
            else:
                regimes.append("unknown")
        else:
            row = item
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if isinstance(row, Sequence):
                weights.append([float(value) for value in row])
                regimes.append("unknown")
            else:
                continue
            steps.append(index)

    weight_array = np.asarray(weights, dtype=np.float64)
    step_array = np.asarray(steps, dtype=np.int64) if have_steps and steps else None
    regime_list = regimes if have_regimes and regimes else None
    return weight_array, step_array, regime_list


def _shade_regime_segments(ax: Any, steps: np.ndarray, regimes: Sequence[str] | None) -> None:
    if not regimes or len(regimes) != len(steps):
        return
    start = 0
    for idx in range(1, len(regimes) + 1):
        if idx == len(regimes) or regimes[idx] != regimes[start]:
            label = str(regimes[start])
            color = REGIME_COLORS.get(label, "#94a3b8")
            ax.axvspan(steps[start], steps[idx - 1], color=color, alpha=0.05, linewidth=0)
            start = idx


def _annotate_regime_trace(ax: Any, steps: np.ndarray, regimes: Sequence[str]) -> None:
    if len(steps) == 0:
        return
    unique: list[str] = []
    for regime in regimes:
        if not unique or unique[-1] != regime:
            unique.append(regime)
    summary = " -> ".join(str(item) for item in unique[:6])
    if len(unique) > 6:
        summary += " -> ..."
    ax.text(
        0.01,
        0.98,
        f"Regime path: {summary}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )


def _policy_payload_to_grid(payload: Mapping[str, Any]) -> dict[str, Any]:
    cells = list(payload.get("cells", []))
    if not cells:
        raise ValueError("policy payload does not contain any cells.")

    x_values = sorted({float(cell["trendGapPct"]) for cell in cells})
    y_values = sorted({float(cell["volatilityPct"]) for cell in cells})
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)

    action_to_index = {label: idx for idx, label in enumerate(ACTION_LABELS)}
    for cell in cells:
        x_val = float(cell["trendGapPct"])
        y_val = float(cell["volatilityPct"])
        action = str(cell.get("bestAction", "flat"))
        if x_val not in x_index or y_val not in y_index:
            continue
        grid[y_index[y_val], x_index[x_val]] = float(action_to_index.get(action, 1))

    return {"x_values": x_values, "y_values": y_values, "grid": grid}


def _coerce_experiment_result(method_name: str, value: Any) -> ExperimentResult:
    if isinstance(value, ExperimentResult):
        return value
    if isinstance(value, Mapping) and "seed_results" in value:
        seed_results = [_coerce_seed_result(item) for item in value.get("seed_results", [])]
        return ExperimentResult(method_name=str(value.get("method_name", method_name)), seed_results=seed_results)
    if isinstance(value, Mapping) and "seedResults" in value:
        seed_results = [_coerce_seed_result(item) for item in value.get("seedResults", [])]
        return ExperimentResult(method_name=str(value.get("method_name", method_name)), seed_results=seed_results)
    if isinstance(value, Sequence) and value and all(isinstance(item, Mapping) for item in value):
        seed_results = [_coerce_seed_result(item) for item in value]
        return ExperimentResult(method_name=method_name, seed_results=seed_results)
    raise TypeError(
        f"Unsupported result payload for '{method_name}'. "
        "Expected ExperimentResult, a list of seed dicts, or a dict with 'seed_results'."
    )


def _coerce_seed_result(value: Mapping[str, Any]) -> SeedResult:
    per_regime = value.get("per_regime", value.get("perRegime", {})) or {}
    expert_utilization = value.get("expert_utilization", value.get("expertUtilization", [])) or None
    extra = value.get("extra", {}) or {}
    return SeedResult(
        seed=int(value.get("seed", 0)),
        cumulative_return=float(value.get("cumulative_return", value.get("strategyReturn", 0.0))),
        sharpe=float(value.get("sharpe", 0.0)),
        sortino=float(value.get("sortino", 0.0)),
        max_drawdown=float(value.get("max_drawdown", value.get("maxDrawdown", 0.0))),
        calmar=float(value.get("calmar", 0.0)),
        win_rate=float(value.get("win_rate", value.get("winRate", 0.0))),
        profit_factor=float(value.get("profit_factor", value.get("profitFactor", 0.0))),
        avg_eval_reward=float(value.get("avg_eval_reward", value.get("avgEvalReward", 0.0))),
        per_regime=dict(per_regime),
        gate_nmi=value.get("gate_nmi", value.get("nmi")),
        gate_ari=value.get("gate_ari", value.get("ari")),
        expert_utilization=list(expert_utilization) if expert_utilization is not None else None,
        extra=dict(extra),
    )
