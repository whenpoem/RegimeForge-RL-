"""Edge-case and numerical stability tests for RegimeForge.

Covers boundary conditions that are not exercised by the standard
functional tests: empty buffers, NaN propagation, corrupted checkpoints,
minimal batch sizes, zero inputs, and single-step episodes.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from backend.regime_lens.config import AgentType, AlgorithmType, GateType, TrainingConfig
from backend.regime_lens.dqn import DQNAgent, PrioritizedReplayBuffer, ReplayBuffer, SumTree
from backend.regime_lens.market import SyntheticMarketEnv
from backend.regime_lens.continuous_agent import (
    ContinuousActorCriticAgent,
    ContinuousReplayBuffer,
    ContinuousPrioritizedReplayBuffer,
)
from backend.regime_lens.continuous_market import ContinuousMarketEnv


class EmptyBufferTests(unittest.TestCase):
    """Verify that agents handle empty buffers gracefully."""

    def test_uniform_buffer_sample_empty_raises(self) -> None:
        buf = ReplayBuffer(capacity=8, observation_dim=3)
        with self.assertRaises(ValueError):
            buf.sample(2, np.random.default_rng(0))

    def test_per_buffer_sample_empty_raises(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=8, observation_dim=3)
        with self.assertRaises(ValueError):
            buf.sample(2, np.random.default_rng(0), beta=0.4)

    def test_continuous_buffer_sample_empty_raises(self) -> None:
        buf = ContinuousReplayBuffer(capacity=8, observation_dim=3, action_dim=2)
        with self.assertRaises(ValueError):
            buf.sample(2, np.random.default_rng(0))

    def test_continuous_per_buffer_sample_empty_raises(self) -> None:
        buf = ContinuousPrioritizedReplayBuffer(capacity=8, observation_dim=3, action_dim=2)
        with self.assertRaises(ValueError):
            buf.sample(2, np.random.default_rng(0), beta=0.4)

    def test_dqn_update_returns_none_on_empty_buffer(self) -> None:
        agent = DQNAgent(
            observation_dim=3, action_dim=2, hidden_dim=8,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=16, batch_size=4,
            device="cpu", seed=0,
        )
        self.assertIsNone(agent.update())

    def test_continuous_sac_update_returns_none_on_empty_buffer(self) -> None:
        config = TrainingConfig(
            algorithm=AlgorithmType.SAC,
            device="cpu",
            autostart=False,
            batch_size=4,
            replay_capacity=16,
            hidden_dim=8,
        )
        agent = ContinuousActorCriticAgent(config, observation_dim=3, action_dim=2)
        self.assertIsNone(agent.update())


class SumTreeEdgeCaseTests(unittest.TestCase):
    def test_single_element_tree(self) -> None:
        tree = SumTree(capacity=1)
        tree.add(1.0)
        self.assertAlmostEqual(tree.total, 1.0)
        self.assertEqual(tree.sample(0.5), 0)

    def test_all_zero_priorities(self) -> None:
        tree = SumTree(capacity=4)
        for _ in range(4):
            tree.add(0.0)
        self.assertAlmostEqual(tree.total, 0.0)


class CorruptedCheckpointTests(unittest.TestCase):
    """Verify that agents reject corrupted checkpoint data."""

    def test_dqn_rejects_mismatched_capacity(self) -> None:
        agent = DQNAgent(
            observation_dim=3, action_dim=2, hidden_dim=8,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=8, batch_size=2,
            device="cpu", seed=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "ckpt"
            agent.save_checkpoint(checkpoint_dir)
            # Corrupt the meta file
            import json
            meta_path = checkpoint_dir / "meta.pt"
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            meta["buffer"]["capacity"] = 999
            torch.save(meta, meta_path)
            loaded = DQNAgent(
                observation_dim=3, action_dim=2, hidden_dim=8,
                learning_rate=1e-3, gamma=0.99, tau=0.01,
                replay_capacity=8, batch_size=2,
                device="cpu", seed=0,
            )
            with self.assertRaises(ValueError):
                loaded.load_checkpoint(checkpoint_dir, weights_only=False)

    def test_sum_tree_rejects_mismatched_capacity(self) -> None:
        tree = SumTree(capacity=4)
        state = tree.state_dict()
        state["capacity"] = 999
        with self.assertRaises(ValueError):
            tree.load_state_dict(state)


class MarketNaNTests(unittest.TestCase):
    """Verify that market environments do not produce NaN under extreme params."""

    def test_extreme_drift_no_nan(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=50,
            warmup_steps=5,
            autostart=False,
            regime_params={
                "bull": {"drift": 0.1, "vol": 0.001, "autocorr": 0.0, "jump_prob": 0.0, "jump_scale": 0.0},
                "bear": {"drift": -0.1, "vol": 0.001, "autocorr": 0.0, "jump_prob": 0.0, "jump_scale": 0.0},
                "chop": {"drift": 0.0, "vol": 0.001, "autocorr": 0.0, "jump_prob": 0.0, "jump_scale": 0.0},
                "shock": {"drift": 0.0, "vol": 0.05, "autocorr": 0.0, "jump_prob": 0.5, "jump_scale": 0.1},
            },
        )
        env = SyntheticMarketEnv(config)
        state = env.reset(seed=42)
        self.assertFalse(np.any(np.isnan(state)), "Initial state contains NaN")
        done = False
        while not done:
            state, _, done, _ = env.step(1)
            self.assertFalse(np.any(np.isnan(state)), f"State at t={env.t} contains NaN")

    def test_high_volatility_no_nan(self) -> None:
        config = TrainingConfig(
            seed=99,
            episode_length=50,
            warmup_steps=5,
            autostart=False,
            regime_params={
                "bull": {"drift": 0.0, "vol": 0.1, "autocorr": 0.0, "jump_prob": 0.0, "jump_scale": 0.0},
                "bear": {"drift": 0.0, "vol": 0.1, "autocorr": 0.0, "jump_prob": 0.0, "jump_scale": 0.0},
                "chop": {"drift": 0.0, "vol": 0.1, "autocorr": 0.0, "jump_prob": 0.0, "jump_scale": 0.0},
                "shock": {"drift": 0.0, "vol": 0.2, "autocorr": 0.0, "jump_prob": 0.3, "jump_scale": 0.1},
            },
        )
        env = SyntheticMarketEnv(config)
        state = env.reset(seed=99)
        self.assertFalse(np.any(np.isnan(state)))
        done = False
        while not done:
            state, _, done, _ = env.step(1)
            self.assertFalse(np.any(np.isnan(state)))


class ZeroInputTests(unittest.TestCase):
    """Verify agent behaviour on all-zero observations."""

    def test_dqn_on_zero_obs(self) -> None:
        agent = DQNAgent(
            observation_dim=7, action_dim=3, hidden_dim=16,
            learning_rate=1e-3, gamma=0.99, tau=0.01,
            replay_capacity=32, batch_size=4,
            device="cpu", seed=0,
        )
        zero_state = np.zeros(7, dtype=np.float32)
        action = agent.select_action(zero_state, epsilon=0.0)
        self.assertIn(action, (0, 1, 2))
        q = agent.q_values(zero_state)
        self.assertEqual(q.shape, (3,))
        self.assertFalse(np.any(np.isnan(q)))

    def test_continuous_ppo_on_zero_obs(self) -> None:
        config = TrainingConfig(
            algorithm=AlgorithmType.PPO,
            device="cpu",
            autostart=False,
            hidden_dim=16,
        )
        agent = ContinuousActorCriticAgent(config, observation_dim=4, action_dim=2)
        zero_state = np.zeros(4, dtype=np.float32)
        result = agent.act(zero_state, deterministic=True)
        action = np.asarray(result["action"])
        self.assertEqual(action.shape, (2,))
        self.assertFalse(np.any(np.isnan(action)))


class SingleStepEpisodeTests(unittest.TestCase):
    """Verify that training works with episode_length=1."""

    def test_dqn_single_step_episode(self) -> None:
        config = TrainingConfig(
            artifact_root=Path(tempfile.mkdtemp()),
            agent_type=AgentType.DQN,
            seed=42,
            episodes=3,
            episode_length=1,
            warmup_steps=1,
            checkpoint_interval=3,
            metrics_flush_interval=3,
            evaluation_episodes=1,
            train_after_steps=0,
            update_every_steps=1,
            gradient_steps=1,
            batch_size=2,
            replay_capacity=16,
            hidden_dim=8,
            device="cpu",
            autostart=False,
        )
        from backend.regime_lens.training import TrainingManager
        manager = TrainingManager(config)
        run_id, _ = manager.store.create_run(manager._config_payload())
        manager._train_loop(run_id)
        summary = manager.store.read_run_summary(run_id)
        self.assertEqual(summary["status"], "completed")


if __name__ == "__main__":
    unittest.main()
