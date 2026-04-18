"""Research analysis toolkit for RCMoE-DQN experiments.

Provides quantitative metrics for evaluating:
1. How well gate weights align with true regime labels (NMI, ARI)
2. How specialised each expert becomes (activation heatmaps)
3. How linearly separable regimes are in hidden representations (probing)
4. Information-theoretic properties of gating decisions (entropy)
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Gate-weight ↔ regime alignment
# ---------------------------------------------------------------------------

def compute_nmi(gate_weights: np.ndarray, true_regimes: list[str]) -> float:
    """Normalised Mutual Information between hard gate assignments and true regimes.

    Args:
        gate_weights: (T, n_experts) soft gate outputs.
        true_regimes: length-T list of regime labels.

    Returns:
        NMI score in [0, 1]. Higher = better alignment.
    """
    hard_assignments = np.argmax(gate_weights, axis=1)
    le = LabelEncoder()
    true_labels = le.fit_transform(true_regimes)
    return float(normalized_mutual_info_score(true_labels, hard_assignments))


def compute_ari(gate_weights: np.ndarray, true_regimes: list[str]) -> float:
    """Adjusted Rand Index between hard gate assignments and true regimes."""
    hard_assignments = np.argmax(gate_weights, axis=1)
    le = LabelEncoder()
    true_labels = le.fit_transform(true_regimes)
    return float(adjusted_rand_score(true_labels, hard_assignments))


# ---------------------------------------------------------------------------
# Expert specialisation
# ---------------------------------------------------------------------------

def expert_activation_matrix(
    gate_weights: np.ndarray,
    true_regimes: list[str],
    regime_labels: tuple[str, ...],
) -> np.ndarray:
    """Build a (n_regimes, n_experts) activation frequency matrix.

    Cell (r, e) = mean gate weight for expert *e* when the true regime
    is *r*.  A diagonal-dominant pattern indicates successful
    specialisation.
    """
    n_experts = gate_weights.shape[1]
    n_regimes = len(regime_labels)
    matrix = np.zeros((n_regimes, n_experts), dtype=np.float64)
    regime_idx = {label: i for i, label in enumerate(regime_labels)}

    for t, regime in enumerate(true_regimes):
        r = regime_idx.get(regime, -1)
        if r < 0:
            continue
        matrix[r] += gate_weights[t]

    # Normalise per regime
    counts = np.zeros(n_regimes, dtype=np.float64)
    for regime in true_regimes:
        r = regime_idx.get(regime, -1)
        if r >= 0:
            counts[r] += 1

    for r in range(n_regimes):
        if counts[r] > 0:
            matrix[r] /= counts[r]

    return matrix


def expert_utilization(gate_weights: np.ndarray) -> np.ndarray:
    """Fraction of time each expert has the highest gate weight.

    Returns shape (n_experts,) summing to 1.
    """
    hard = np.argmax(gate_weights, axis=1)
    n_experts = gate_weights.shape[1]
    counts = np.bincount(hard, minlength=n_experts).astype(np.float64)
    return counts / (counts.sum() + 1e-10)


def specialisation_score(activation_matrix: np.ndarray) -> float:
    """Scalar specialisation score: how diagonal-dominant the matrix is.

    1.0 = perfect diagonal, 0.0 = uniform (no specialisation).
    Uses the ratio of diagonal sum to total sum, normalised.
    """
    n = min(activation_matrix.shape)
    if n == 0:
        return 0.0
    diag = sum(activation_matrix[i, i] for i in range(n))
    total = float(activation_matrix.sum())
    if total < 1e-10:
        return 0.0
    # Chance-level diagonal is n_regimes * (1/n_experts) ≈ 1.0 when square
    chance = n * (1.0 / activation_matrix.shape[1])
    return float((diag - chance) / (total - chance + 1e-10))


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------

def linear_probing_accuracy(
    hidden_states: np.ndarray,
    true_regimes: list[str],
    max_iter: int = 500,
) -> float:
    """Train a linear classifier on hidden representations to predict regime.

    This measures how linearly separable the regime information is in the
    agent's learned representation space.  Higher accuracy → the network
    has implicitly encoded regime structure.
    """
    le = LabelEncoder()
    labels = le.fit_transform(true_regimes)
    n_classes = len(le.classes_)

    if len(hidden_states) < n_classes * 2:
        return 0.0

    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )
    clf.fit(hidden_states, labels)
    predictions = clf.predict(hidden_states)
    return float(accuracy_score(labels, predictions))


# ---------------------------------------------------------------------------
# Representation quality
# ---------------------------------------------------------------------------

def regime_separation_score(
    hidden_states: np.ndarray,
    true_regimes: list[str],
) -> float:
    """Measure how well-separated regime clusters are in representation space.

    Uses the ratio of between-cluster variance to total variance
    (similar to the Calinski-Harabasz index idea).
    """
    le = LabelEncoder()
    labels = le.fit_transform(true_regimes)
    n_classes = len(le.classes_)

    global_mean = np.mean(hidden_states, axis=0)
    total_var = float(np.sum((hidden_states - global_mean) ** 2))  # SST

    between_var = 0.0
    for c in range(n_classes):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        class_mean = np.mean(hidden_states[mask], axis=0)
        between_var += n_c * float(np.sum((class_mean - global_mean) ** 2))

    return float(between_var / (total_var + 1e-10))


# ---------------------------------------------------------------------------
# Information-theoretic measures
# ---------------------------------------------------------------------------

def gate_entropy(gate_weights: np.ndarray) -> float:
    """Mean entropy of gate-weight distributions.

    Low entropy = confident routing (one expert dominates).
    High entropy = uncertain / uniform routing.
    """
    # Clamp to avoid log(0)
    w = np.clip(gate_weights, 1e-10, 1.0)
    per_step_entropy = -np.sum(w * np.log(w), axis=1)
    return float(np.mean(per_step_entropy))


def gate_entropy_per_regime(
    gate_weights: np.ndarray,
    true_regimes: list[str],
    regime_labels: tuple[str, ...],
) -> dict[str, float]:
    """Mean gate entropy broken down by true regime."""
    w = np.clip(gate_weights, 1e-10, 1.0)
    per_step_entropy = -np.sum(w * np.log(w), axis=1)
    result: dict[str, float] = {}
    for label in regime_labels:
        mask = np.array([r == label for r in true_regimes], dtype=bool)
        if mask.sum() > 0:
            result[label] = float(np.mean(per_step_entropy[mask]))
    return result


# ---------------------------------------------------------------------------
# Full analysis bundle
# ---------------------------------------------------------------------------

def full_regime_analysis(
    gate_weights: np.ndarray,
    true_regimes: list[str],
    regime_labels: tuple[str, ...],
    hidden_states: np.ndarray | None = None,
) -> dict:
    """Run all analysis metrics and return a JSON-serialisable bundle."""
    act_matrix = expert_activation_matrix(gate_weights, true_regimes, regime_labels)
    result = {
        "nmi": compute_nmi(gate_weights, true_regimes),
        "ari": compute_ari(gate_weights, true_regimes),
        "gate_entropy": gate_entropy(gate_weights),
        "gate_entropy_per_regime": gate_entropy_per_regime(gate_weights, true_regimes, regime_labels),
        "expert_utilization": expert_utilization(gate_weights).tolist(),
        "activation_matrix": act_matrix.tolist(),
        "specialisation_score": specialisation_score(act_matrix),
    }
    if hidden_states is not None and len(hidden_states) > 0:
        result["probing_accuracy"] = linear_probing_accuracy(hidden_states, true_regimes)
        result["separation_score"] = regime_separation_score(hidden_states, true_regimes)
    return result
