from __future__ import annotations

import unittest

import numpy as np

from backend.regime_lens.config import TrainingConfig
from backend.regime_lens.continuous_market import ContinuousMarketEnv


class ContinuousMarketEnvTests(unittest.TestCase):
    def test_entry_price_preserves_basis_across_same_side_resizes(self) -> None:
        config = TrainingConfig(
            continuous_actions=True,
            real_data_symbols=("SPY",),
            episode_length=6,
            warmup_steps=4,
            seed=21,
            autostart=False,
        )
        env = ContinuousMarketEnv(config)
        env.reset(seed=21)

        initial_entry_price = float(env.prices[env.t, 0])
        env.step(np.array([0.4], dtype=np.float32))
        self.assertAlmostEqual(float(env.entry_prices[0]), initial_entry_price)

        scale_in_price = float(env.prices[env.t, 0])
        env.step(np.array([0.8], dtype=np.float32))
        expected_weighted_entry = ((0.4 * initial_entry_price) + (0.4 * scale_in_price)) / 0.8
        self.assertAlmostEqual(float(env.entry_prices[0]), expected_weighted_entry)

        env.step(np.array([0.3], dtype=np.float32))
        self.assertAlmostEqual(float(env.entry_prices[0]), expected_weighted_entry)

        flip_price = float(env.prices[env.t, 0])
        env.step(np.array([-0.2], dtype=np.float32))
        self.assertAlmostEqual(float(env.entry_prices[0]), flip_price)

        env.step(np.array([0.0], dtype=np.float32))
        self.assertTrue(np.isnan(env.entry_prices[0]))


if __name__ == "__main__":
    unittest.main()
