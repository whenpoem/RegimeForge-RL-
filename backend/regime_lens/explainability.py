from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "online", getattr(model, "model", model))


def _model_device(model: Any) -> torch.device:
    module = _unwrap_model(model)
    if isinstance(module, torch.nn.Module):
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return torch.device("cpu")


def _ensure_tensor(observation: np.ndarray | torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, bool]:
    if isinstance(observation, torch.Tensor):
        tensor = observation.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(observation, device=device, dtype=torch.float32)
    squeezed = tensor.ndim == 1
    if squeezed:
        tensor = tensor.unsqueeze(0)
    return tensor, squeezed


def _forward_outputs(model: Any, observation: torch.Tensor) -> dict[str, torch.Tensor]:
    network = _unwrap_model(model)
    if hasattr(network, "expert_outputs"):
        outputs = network.expert_outputs(observation)
        if "mean" not in outputs and callable(network):
            forward_outputs = network(observation)
            if isinstance(forward_outputs, dict):
                outputs = {**outputs, **forward_outputs}
        return outputs

    result = network(observation)
    if isinstance(result, tuple) and len(result) == 2:
        return {"primary": result[0], "gate_weights": result[1]}
    if isinstance(result, dict):
        return result
    if isinstance(result, torch.Tensor):
        return {"primary": result}
    raise TypeError(f"Unsupported model output type for explainability: {type(result)!r}")


def gate_attribution(
    model: Any,
    observation: np.ndarray | torch.Tensor,
    *,
    baseline: np.ndarray | torch.Tensor | None = None,
    steps: int = 16,
    expert_index: int | None = None,
) -> dict[str, Any]:
    device = _model_device(model)
    obs, squeezed = _ensure_tensor(observation, device=device)
    base, _ = _ensure_tensor(np.zeros_like(obs.cpu().numpy()) if baseline is None else baseline, device=device)

    latest_gate: torch.Tensor | None = None
    integrated_gradients = torch.zeros_like(obs)
    alphas = torch.linspace(0.0, 1.0, max(steps, 2), device=device)

    for alpha in alphas:
        sample = (base + alpha * (obs - base)).detach().requires_grad_(True)
        outputs = _forward_outputs(model, sample)
        gate_weights = outputs.get("gate_weights")
        if gate_weights is None:
            raise ValueError("Model does not expose gate weights for attribution.")
        latest_gate = gate_weights.detach()
        target_expert = int(gate_weights[0].argmax().item()) if expert_index is None else int(expert_index)
        target = gate_weights[:, target_expert].sum()
        gradients = torch.autograd.grad(target, sample, allow_unused=False)[0]
        integrated_gradients = integrated_gradients + gradients

    attribution = (obs - base) * (integrated_gradients / len(alphas))
    if squeezed:
        attribution = attribution.squeeze(0)
        latest_gate = latest_gate.squeeze(0) if latest_gate is not None else None

    return {
        "gate_weights": None if latest_gate is None else latest_gate.detach().cpu().numpy(),
        "target_expert": None if latest_gate is None else int(np.argmax(latest_gate.detach().cpu().numpy())),
        "attribution": attribution.detach().cpu().numpy(),
    }


def expert_counterfactual(
    model: Any,
    observation: np.ndarray | torch.Tensor,
    *,
    expert_index: int,
) -> dict[str, Any]:
    device = _model_device(model)
    obs, squeezed = _ensure_tensor(observation, device=device)
    outputs = _forward_outputs(model, obs)

    if "expert_means" in outputs and "gate_weights" in outputs:
        gate = outputs["gate_weights"]
        expert_means = outputs["expert_means"]
        mixed_action = outputs.get("mean")
        if mixed_action is None:
            mixed_action = (gate.unsqueeze(-1) * expert_means).sum(dim=1)
        counterfactual = expert_means[:, expert_index]
        payload = {
            "gate_weights": gate,
            "mixed_action": torch.tanh(mixed_action),
            "counterfactual_action": torch.tanh(counterfactual),
            "delta": torch.tanh(counterfactual) - torch.tanh(mixed_action),
        }
        if "expert_values" in outputs:
            payload["mixed_value"] = outputs.get("value", (gate * outputs["expert_values"]).sum(dim=1, keepdim=True))
            payload["counterfactual_value"] = outputs["expert_values"][:, expert_index : expert_index + 1]
        return {
            key: value.squeeze(0).detach().cpu().numpy() if squeezed else value.detach().cpu().numpy()
            for key, value in payload.items()
        }

    network = _unwrap_model(model)
    if hasattr(network, "expert_q_values"):
        mixed_q, gate = network(obs)
        expert_q = network.expert_q_values(obs)[:, expert_index]
        payload = {
            "gate_weights": gate,
            "mixed_q": mixed_q,
            "counterfactual_q": expert_q,
            "mixed_action": mixed_q.argmax(dim=-1),
            "counterfactual_action": expert_q.argmax(dim=-1),
        }
        return {
            key: value.squeeze(0).detach().cpu().numpy() if squeezed and value.ndim > 0 else value.detach().cpu().numpy()
            for key, value in payload.items()
        }

    raise ValueError("Model does not expose expert-level outputs for counterfactual analysis.")


def decision_boundary(
    model: Any,
    baseline_state: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    feature_x: int,
    feature_y: int,
    span: float = 2.0,
    grid_size: int = 21,
    target: str = "policy",
) -> dict[str, Any]:
    base = np.asarray(baseline_state, dtype=np.float32).copy()
    x_values = np.linspace(base[feature_x] - span, base[feature_x] + span, grid_size, dtype=np.float32)
    y_values = np.linspace(base[feature_y] - span, base[feature_y] + span, grid_size, dtype=np.float32)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    boundary_mask = np.zeros((grid_size, grid_size), dtype=bool)

    previous_label: float | None = None
    for y_index, y_value in enumerate(y_values):
        row_labels: list[float] = []
        for x_index, x_value in enumerate(x_values):
            candidate = base.copy()
            candidate[feature_x] = x_value
            candidate[feature_y] = y_value
            row_labels.append(_decision_value(model, candidate, target=target))
            grid[y_index, x_index] = row_labels[-1]

        for x_index, label in enumerate(row_labels):
            left = row_labels[x_index - 1] if x_index > 0 else label
            boundary_mask[y_index, x_index] = bool(label != left or (previous_label is not None and label != previous_label))
            previous_label = label

    return {
        "x_values": x_values,
        "y_values": y_values,
        "decision_grid": grid,
        "boundary_mask": boundary_mask,
        "feature_x": feature_x,
        "feature_y": feature_y,
        "target": target,
    }


def _decision_value(model: Any, state: np.ndarray, *, target: str) -> float:
    device = _model_device(model)
    obs, _ = _ensure_tensor(state, device=device)
    outputs = _forward_outputs(model, obs)
    decision_target = target.strip().lower()

    if decision_target == "gate":
        gate_weights = outputs.get("gate_weights")
        if gate_weights is None:
            raise ValueError("Model does not expose gate weights for gate decision boundaries.")
        return float(gate_weights.argmax(dim=-1).item())

    if "primary" in outputs:
        primary = outputs["primary"]
        return float(primary.argmax(dim=-1).item())
    if "mean" in outputs:
        return float(torch.tanh(outputs["mean"])[0, 0].item())

    network = _unwrap_model(model)
    if hasattr(network, "act"):
        acted = network.act(obs, deterministic=True)
        action = acted["action"]
        if isinstance(action, torch.Tensor):
            return float(action.reshape(-1)[0].item())
        return float(np.asarray(action).reshape(-1)[0])

    raise ValueError("Model does not expose a supported policy or Q-value surface.")


def find_transition_points(labels: Sequence[Any]) -> np.ndarray:
    array = np.asarray(labels)
    if array.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(array[1:] != array[:-1]) + 1


def transition_lag(
    true_regimes: Sequence[Any],
    inferred_regimes: Sequence[Any],
    *,
    max_lag: int = 12,
) -> dict[str, Any]:
    true_array = np.asarray(true_regimes)
    inferred_array = np.asarray(inferred_regimes)
    length = min(true_array.size, inferred_array.size)
    true_array = true_array[:length]
    inferred_array = inferred_array[:length]

    true_transitions = find_transition_points(true_array)
    inferred_transitions = find_transition_points(inferred_array)
    matched_lags: list[int] = []
    for transition in true_transitions:
        if inferred_transitions.size == 0:
            break
        nearest_index = int(np.argmin(np.abs(inferred_transitions - transition)))
        matched_lags.append(int(inferred_transitions[nearest_index] - transition))

    lag_scores: dict[int, float] = {}
    best_lag = 0
    best_score = float("-inf")
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aligned_true = true_array[-lag:]
            aligned_inferred = inferred_array[: length + lag]
        elif lag > 0:
            aligned_true = true_array[: length - lag]
            aligned_inferred = inferred_array[lag:]
        else:
            aligned_true = true_array
            aligned_inferred = inferred_array

        score = float(np.mean(aligned_true == aligned_inferred)) if aligned_true.size else 0.0
        lag_scores[lag] = score
        if score > best_score:
            best_lag = lag
            best_score = score

    lag_array = np.asarray(matched_lags, dtype=np.float32)
    return {
        "true_transitions": true_transitions,
        "inferred_transitions": inferred_transitions,
        "matched_lags": lag_array,
        "mean_lag": float(lag_array.mean()) if lag_array.size else 0.0,
        "median_lag": float(np.median(lag_array)) if lag_array.size else 0.0,
        "best_global_lag": best_lag,
        "best_global_accuracy": best_score,
        "lag_scores": lag_scores,
    }


__all__ = [
    "decision_boundary",
    "expert_counterfactual",
    "find_transition_points",
    "gate_attribution",
    "transition_lag",
]
