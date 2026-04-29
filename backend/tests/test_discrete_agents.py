"""Unit tests for discrete RL agents: DQN, RCMoE, Oracle, HMM.

Covers update loops, gate weights, observation augmentation, checkpoint
round-trips, replay buffer state dicts, and NoisyLinear behaviour.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from backend.regime_lens.config import AgentType, GateType, TrainingConfig, REGIME_LABELS
from backend.regime_lens.dqn import (
    DQNAgent,
    NoisyLinear,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    SumTree,
)
from backend.regime_lens.hmm_dqn import HMMDQNAgent, RegimeDetector
from backend.regime_lens.oracle_dqn import OracleDQNAgent
from backend.regime_lens.rcmoe import RCMoEAgent


class ReplayBufferTests(unittest.TestCase):
    def test_add_and_sample(self) -> None:
        buf = ReplayBuffer(capacity=8, observation_dim=3)
        rng = np.random.default_rng(42)
        for i in range(5):
            buf.add(np.ones(3, dtype=np.float32) * i, i % 2, float(i), np.ones(3, dtype=np.float32) * (i + 1), False)
        self.assertEqual(len(buf), 5)
        states, actions, rewards, next_states, dones = buf.sample(3, rng)
        self.assertEqual(states.shape, (3, 3))
        self.assertEqual(actions.shape, (3,))
        self.assertEqual(rewards.shape, (3,))

    def test_state_dict_round_trip(self) -> None:
        buf = ReplayBuffer(capacity=8, observation_dim=3)
        for i in range(4):
            buf.add(np.ones(3, dtype=np.float32) * i, i % 2, float(i), np.ones(3, dtype=np.float32) * (i + 1), False)
        state = buf.state_dict()
        restored = ReplayBuffer(capacity=8, observation_dim=3)
        restored.load_state_dict(state)
        self.assertEqual(restored.size, buf.size)
        np.testing.assert_array_equal(restored.states[:buf.size], buf.states[:buf.size])


class PrioritizedReplayBufferTests(unittest.TestCase):
    def test_add_sample_and_update(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8, observation_dim=3)
        rng = np.random.default_rng(42)
        for i in range(6):
            buf.add(np.ones(3, dtype=np.float32) * i, i % 2, float(i), np.ones(3, dtype=np.float32) * (i + 1), False)
        states, actions, rewards, next_states, dones, weights, indices = buf.sample(4, rng, beta=0.4)
        self.assertEqual(states.shape, (4, 3))
        self.assertEqual(weights.shape, (4,))
        self.assertTrue(np.all(weights > 0))
        # Update priorities
        buf.update_priorities(indices, np.ones(4) * 2.0)

    def test_state_dict_round_trip(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8, observation_dim=3)
        for i in range(4):
            buf.add(np.ones(3, dtype=np.float32) * i, i % 2, float(i), np.ones(3, dtype=np.float32) * (i + 1), False)
        state = buf.state_dict()
        restored = PrioritizedReplayBuffer(capacity=8, observation_dim=3)
        restored.load_state_dict(state)
        self.assertEqual(restored.size, buf.size)
        np.testing.assert_array_equal(restored.tree.tree, buf.tree.tree)


class NoisyLinearTests(unittest.TestCase):
    def test_noise_reset_changes_epsilon(self) -> None:
        layer = NoisyLinear(8, 4)
        eps_before = layer.weight_epsilon.clone()
        layer.reset_noise()
        eps_after = layer.weight_epsilon.clone()
        # Very unlikely to be exactly the same after reset
        self.assertFalse(torch.equal(eps_before, eps_after))

    def test_eval_mode_uses_mean_only(self) -> None:
        layer = NoisyLinear(8, 4)
        layer.eval()
        x = torch.randn(2, 8)
        out1 = layer(x)
        out2 = layer(x)
        torch.testing.assert_close(out1, out2)


class DQNAgentTests(unittest.TestCase):
    def test_full_update_cycle(self) -> None:
        agent = DQNAgent(
            observation_dim=4, action_dim=3, hidden_dim=16,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=42, use_per=True,
        )
        rng = np.random.default_rng(42)
        for i in range(12):
            s = rng.standard_normal(4).astype(np.float32)
            ns = rng.standard_normal(4).astype(np.float32)
            agent.store(s, i % 3, float(i) * 0.1, ns, i == 11)
        loss = agent.update(per_beta=0.5)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_checkpoint_round_trip(self) -> None:
        agent = DQNAgent(
            observation_dim=4, action_dim=3, hidden_dim=16,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=42, use_per=True,
        )
        rng = np.random.default_rng(42)
        for i in range(8):
            s = rng.standard_normal(4).astype(np.float32)
            ns = rng.standard_normal(4).astype(np.float32)
            agent.store(s, i % 3, float(i) * 0.1, ns, i == 7)
        agent.update(per_beta=0.5)

        ref_state = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        expected_q = agent.q_values(ref_state)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            agent.save_checkpoint(ckpt)
            restored = DQNAgent(
                observation_dim=4, action_dim=3, hidden_dim=16,
                learning_rate=1e-3, gamma=0.99, tau=0.01,
                replay_capacity=32, batch_size=4,
                device="cpu", seed=42, use_per=True,
            )
            restored.load_checkpoint(ckpt, weights_only=False)
            restored_q = restored.q_values(ref_state)
            np.testing.assert_allclose(restored_q, expected_q, atol=1e-5)


class RCMoEAgentTests(unittest.TestCase):
    def test_gate_weights_shape_and_normalization(self) -> None:
        agent = RCMoEAgent(
            observation_dim=7, action_dim=3, hidden_dim=16,
            n_experts=4, gate_hidden_dim=16,
            load_balance_weight=0.01,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=42,
        )
        state = np.random.default_rng(42).standard_normal(7).astype(np.float32)
        weights = agent.gate_weights(state)
        self.assertEqual(weights.shape, (4,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))

    def test_update_with_per(self) -> None:
        agent = RCMoEAgent(
            observation_dim=7, action_dim=3, hidden_dim=16,
            n_experts=3, gate_hidden_dim=8,
            load_balance_weight=0.01,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=42, use_per=True,
        )
        rng = np.random.default_rng(42)
        for i in range(12):
            s = rng.standard_normal(7).astype(np.float32)
            ns = rng.standard_normal(7).astype(np.float32)
            agent.store(s, i % 3, float(i) * 0.1, ns, i == 11)
        loss = agent.update(per_beta=0.5)
        self.assertIsNotNone(loss)

    def test_temporal_gate(self) -> None:
        # For temporal gate, observation_dim = base_dim * context_len = 7 * 4 = 28
        agent = RCMoEAgent(
            observation_dim=28, action_dim=3, hidden_dim=16,
            n_experts=3, gate_hidden_dim=8,
            load_balance_weight=0.01,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=42,
            gate_type=GateType.TEMPORAL, context_len=4,
        )
        context = np.random.default_rng(42).standard_normal(28).astype(np.float32)
        weights = agent.gate_weights(context)
        self.assertEqual(weights.shape, (3,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)


class OracleDQNAgentTests(unittest.TestCase):
    def test_augment_state(self) -> None:
        agent = OracleDQNAgent(
            base_observation_dim=7, action_dim=3, hidden_dim=16,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=42,
        )
        base = np.ones(7, dtype=np.float32)
        augmented = agent.augment_state(base, "bull")
        self.assertEqual(augmented.shape, (11,))  # 7 + 4 regimes
        self.assertAlmostEqual(float(augmented[7]), 1.0)  # bull index 0
        self.assertAlmostEqual(float(augmented[8]), 0.0)  # bear

    def test_unknown_regime(self) -> None:
        agent = OracleDQNAgent(
            base_observation_dim=5, action_dim=2, hidden_dim=8,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=16, batch_size=2,
            device="cpu", seed=42,
        )
        base = np.zeros(5, dtype=np.float32)
        augmented = agent.augment_state(base, "nonexistent")
        self.assertEqual(augmented.shape, (9,))
        self.assertTrue(np.all(augmented[5:] == 0))


class HMMDQNAgentTests(unittest.TestCase):
    def test_detector_fit_predict(self) -> None:
        detector = RegimeDetector(n_components=3, window_size=5, seed=42)
        returns = np.random.default_rng(42).normal(0, 0.01, 100)
        detector.fit(returns)
        labels = detector.predict(returns)
        self.assertEqual(len(labels), 96)  # 100 - 5 + 1
        probs = detector.predict_proba(returns)
        self.assertEqual(probs.shape, (96, 3))

    def test_augment_state(self) -> None:
        agent = HMMDQNAgent(
            base_observation_dim=7, action_dim=3, hidden_dim=16,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=42,
        )
        returns = np.random.default_rng(42).normal(0, 0.01, 50)
        agent.fit_detector(returns)
        base = np.ones(7, dtype=np.float32)
        augmented = agent.augment_state(base, returns[-10:])
        self.assertEqual(augmented.shape, (11,))  # 7 + 4 components

    def test_checkpoint_round_trip(self) -> None:
        agent = HMMDQNAgent(
            base_observation_dim=7, action_dim=3, hidden_dim=16,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=42,
        )
        returns = np.random.default_rng(42).normal(0, 0.01, 50)
        agent.fit_detector(returns)
        base = np.ones(7, dtype=np.float32)
        expected = agent.augment_state(base, returns[-10:])

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            agent.save_checkpoint(ckpt)
            restored = HMMDQNAgent(
                base_observation_dim=7, action_dim=3, hidden_dim=16,
                learning_rate=1e-3, gamma=0.99, tau=0.01,
                replay_capacity=32, batch_size=4,
                device="cpu", seed=42,
            )
            restored.load_checkpoint(ckpt, weights_only=False)
            restored_state = restored.augment_state(base, returns[-10:])
            np.testing.assert_allclose(restored_state, expected)


if __name__ == "__main__":
    unittest.main()
