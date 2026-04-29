"""Tests for analysis, explainability, and metrics modules."""

from __future__ import annotations

import unittest

import numpy as np

from backend.regime_lens.analysis import (
    compute_nmi,
    compute_ari,
    expert_activation_matrix,
    expert_utilization,
)
from backend.regime_lens.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    profit_factor,
    per_regime_metrics,
    episode_metrics,
)
from backend.regime_lens.explainability import (
    decision_boundary,
    expert_counterfactual,
    find_transition_points,
    gate_attribution,
    transition_lag,
)
from backend.regime_lens.config import REGIME_LABELS


class NMITests(unittest.TestCase):
    def test_perfect_alignment(self) -> None:
        """Gate assigns each regime to a unique expert."""
        n_steps = 120
        n_experts = 4
        gate = np.zeros((n_steps, n_experts), dtype=np.float64)
        regimes = []
        for i, label in enumerate(REGIME_LABELS):
            for _ in range(30):
                gate[len(regimes), i] = 1.0
                regimes.append(label)
        nmi = compute_nmi(gate, regimes)
        self.assertAlmostEqual(nmi, 1.0, places=4)

    def test_random_alignment(self) -> None:
        """Random gate weights should give low NMI."""
        rng = np.random.default_rng(42)
        gate = rng.random((200, 4))
        gate /= gate.sum(axis=1, keepdims=True)
        regimes = [REGIME_LABELS[i % 4] for i in range(200)]
        nmi = compute_nmi(gate, regimes)
        self.assertLess(nmi, 0.1)


class ARITests(unittest.TestCase):
    def test_perfect_alignment(self) -> None:
        n_steps = 120
        n_experts = 4
        gate = np.zeros((n_steps, n_experts), dtype=np.float64)
        regimes = []
        for i, label in enumerate(REGIME_LABELS):
            for _ in range(30):
                gate[len(regimes), i] = 1.0
                regimes.append(label)
        ari = compute_ari(gate, regimes)
        self.assertAlmostEqual(ari, 1.0, places=4)


class ExpertActivationMatrixTests(unittest.TestCase):
    def test_shape(self) -> None:
        gate = np.ones((40, 3), dtype=np.float64) / 3.0
        regimes = [REGIME_LABELS[i % 4] for i in range(40)]
        matrix = expert_activation_matrix(gate, regimes, REGIME_LABELS)
        self.assertEqual(matrix.shape, (4, 3))

    def test_diagonal_dominant(self) -> None:
        n = 120
        gate = np.zeros((n, 4), dtype=np.float64)
        regimes = []
        for i, label in enumerate(REGIME_LABELS):
            for _ in range(30):
                gate[len(regimes), i] = 1.0
                regimes.append(label)
        matrix = expert_activation_matrix(gate, regimes, REGIME_LABELS)
        for i in range(4):
            self.assertGreater(matrix[i, i], 0.9)


class ExpertUtilizationTests(unittest.TestCase) :
    def test_uniform_utilization(self) -> None:
        rng = np.random.default_rng(42)
        gate = rng.random((1000, 4))
        gate /= gate.sum(axis=1, keepdims=True)
        util = expert_utilization(gate)
        self.assertEqual(util.shape, (4,))
        self.assertAlmostEqual(float(util.sum()), 1.0, places=5)
        for u in util:
            self.assertGreater(u, 0.15)


class MetricsEdgeCaseTests(unittest.TestCase):
    def test_sharpe_empty(self) -> None:
        self.assertEqual(sharpe_ratio(np.array([])), 0.0)

    def test_sharpe_all_positive(self) -> None:
        returns = np.ones(100) * 0.001
        self.assertGreater(sharpe_ratio(returns), 0)

    def test_sortino_empty(self) -> None:
        self.assertEqual(sortino_ratio(np.array([])), 0.0)

    def test_max_drawdown_empty(self) -> None:
        self.assertEqual(max_drawdown(np.array([])), 0.0)

    def test_max_drawdown_no_drawdown(self) -> None:
        cumulative = np.linspace(0, 0.5, 100)
        self.assertAlmostEqual(max_drawdown(cumulative), 0.0, places=5)

    def test_win_rate_empty(self) -> None:
        self.assertEqual(win_rate(np.array([])), 0.0)

    def test_profit_factor_all_positive(self) -> None:
        pnls = np.ones(10) * 0.01
        self.assertGreater(profit_factor(pnls), 100)

    def test_per_regime_metrics(self) -> None:
        returns = np.random.default_rng(42).normal(0, 0.01, 100)
        regimes = [REGIME_LABELS[i % 4] for i in range(100)]
        result = per_regime_metrics(returns, regimes, REGIME_LABELS)
        self.assertEqual(len(result), 4)
        for label in REGIME_LABELS:
            self.assertIn(label, result)
            self.assertIn("mean_return", result[label])
            self.assertIn("sharpe", result[label])

    def test_episode_metrics(self) -> None:
        pnls = np.random.default_rng(42).normal(0, 0.01, 50)
        regimes = [REGIME_LABELS[i % 4] for i in range(50)]
        result = episode_metrics(pnls, regimes, REGIME_LABELS)
        self.assertIn("cumulative_return", result)
        self.assertIn("sharpe", result)
        self.assertIn("per_regime", result)


class TransitionLagTests(unittest.TestCase):
    def test_perfect_sync(self) -> None:
        true = ["bull"] * 10 + ["bear"] * 10
        inferred = ["bull"] * 10 + ["bear"] * 10
        result = transition_lag(true, inferred)
        self.assertAlmostEqual(result["mean_lag"], 0.0)

    def test_lag_of_one(self) -> None:
        true = ["bull"] * 10 + ["bear"] * 10
        inferred = ["bull"] * 11 + ["bear"] * 9
        result = transition_lag(true, inferred)
        self.assertGreater(abs(result["mean_lag"]), 0)


class FindTransitionPointsTests(unittest.TestCase):
    def test_no_transitions(self) -> None:
        result = find_transition_points(["bull"] * 10)
        self.assertEqual(len(result), 0)

    def test_one_transition(self) -> None:
        result = find_transition_points(["bull"] * 5 + ["bear"] * 5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 5)


if __name__ == "__main__":
    unittest.main()
