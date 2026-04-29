"""Tests for synthetic market environments."""

from __future__ import annotations

import unittest

import numpy as np

from backend.regime_lens.config import REGIME_LABELS, TrainingConfig
from backend.regime_lens.market import SyntheticMarketEnv
from backend.regime_lens.continuous_market import (
    ContinuousMarketEnv,
    MultiAssetContinuousMarketEnv,
)


class SyntheticMarketEnvTests(unittest.TestCase):
    def test_reset_produces_correct_shapes(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=50,
            warmup_steps=10,
            autostart=False,
        )
        env = SyntheticMarketEnv(config)
        state = env.reset(seed=42)
        self.assertEqual(state.shape, (7,))
        self.assertEqual(len(env.prices), 50 + 10 + 2)  # returns + initial price
        self.assertEqual(len(env.regimes), 50 + 10 + 1)

    def test_different_seeds_different_trajectories(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=50,
            warmup_steps=10,
            autostart=False,
        )
        env1 = SyntheticMarketEnv(config)
        s1 = env1.reset(seed=42)
        env2 = SyntheticMarketEnv(config)
        s2 = env2.reset(seed=99)
        self.assertFalse(np.allclose(env1.prices, env2.prices))

    def test_regime_schedule_override(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=10,
            warmup_steps=2,
            autostart=False,
        )
        env = SyntheticMarketEnv(config)
        schedule = ["bull"] * 6 + ["bear"] * 7  # total_steps = 13
        env.set_regime_schedule(schedule)
        env.reset(seed=42)
        self.assertEqual(env.regimes[:6], ["bull"] * 6)
        self.assertEqual(env.regimes[6:13], ["bear"] * 7)

    def test_step_updates_position(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=20,
            warmup_steps=5,
            autostart=False,
        )
        env = SyntheticMarketEnv(config)
        env.reset(seed=42)
        # ACTION_VALUES = (-1, 0, 1) → step(2) = long
        state, reward, done, info = env.step(2)  # long
        self.assertEqual(env.position, 1)
        self.assertFalse(done)
        self.assertIn("regime", info)

    def test_oracle_observation(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=10,
            warmup_steps=2,
            autostart=False,
        )
        env = SyntheticMarketEnv(config)
        env.reset(seed=42)
        oracle = env.observe_oracle()
        self.assertEqual(oracle.state.shape, (11,))  # 7 features + 4 regime one-hot
        # One-hot should sum to 1
        self.assertAlmostEqual(float(oracle.state[7:].sum()), 1.0)


class ContinuousMarketEnvTests(unittest.TestCase):
    def test_single_asset_shapes(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=20,
            warmup_steps=5,
            continuous_actions=True,
            autostart=False,
        )
        env = ContinuousMarketEnv(config)
        state = env.reset(seed=42)
        self.assertEqual(state.shape, (7,))
        self.assertEqual(env.action_dim, 1)

    def test_step_with_continuous_action(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=20,
            warmup_steps=5,
            continuous_actions=True,
            autostart=False,
        )
        env = ContinuousMarketEnv(config)
        env.reset(seed=42)
        state, reward, done, info = env.step(np.array([0.5]))
        self.assertEqual(state.shape, (7,))
        self.assertAlmostEqual(float(env.positions[0]), 0.5)

    def test_multi_asset_shapes(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=20,
            warmup_steps=5,
            continuous_actions=True,
            real_data_symbols=("SPY", "QQQ", "GLD"),
            autostart=False,
        )
        env = MultiAssetContinuousMarketEnv(config)
        state = env.reset(seed=42)
        self.assertEqual(state.shape, (21,))  # 7 * 3
        self.assertEqual(env.action_dim, 3)

    def test_oracle_observation_continuous(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=10,
            warmup_steps=2,
            continuous_actions=True,
            autostart=False,
        )
        env = ContinuousMarketEnv(config)
        env.reset(seed=42)
        oracle = env.observe_oracle()
        self.assertEqual(oracle.state.shape, (11,))  # 7 + 4 regimes

    def test_nonstationary_drift_mode(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=50,
            warmup_steps=5,
            continuous_actions=True,
            nonstationary_mode="drift",
            nonstationary_drift_scale=0.5,
            autostart=False,
        )
        env = ContinuousMarketEnv(config)
        state = env.reset(seed=42)
        self.assertFalse(np.any(np.isnan(state)))
        done = False
        while not done:
            state, _, done, _ = env.step(np.array([0.3]))
            self.assertFalse(np.any(np.isnan(state)))

    def test_nonstationary_volatility_mode(self) -> None:
        config = TrainingConfig(
            seed=42,
            episode_length=50,
            warmup_steps=5,
            continuous_actions=True,
            nonstationary_mode="volatility",
            nonstationary_drift_scale=0.3,
            autostart=False,
        )
        env = ContinuousMarketEnv(config)
        state = env.reset(seed=42)
        self.assertFalse(np.any(np.isnan(state)))


if __name__ == "__main__":
    unittest.main()
