from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from backend.regime_lens.actor_critic import (
    PPOActorCritic,
    RCMoEActorCritic,
    SACActorCritic,
    build_actor_critic,
)
from backend.regime_lens.config import AgentType, AlgorithmType, TrainingConfig
from backend.regime_lens.continuous_market import (
    ContinuousMarketEnv,
    MultiAssetContinuousMarketEnv,
    make_market_env,
)
from backend.regime_lens.explainability import (
    decision_boundary,
    expert_counterfactual,
    gate_attribution,
    transition_lag,
)
from backend.regime_lens.market import SyntheticMarketEnv


class Batch34ScaffoldingTests(unittest.TestCase):
    def test_make_market_env_preserves_discrete_default(self) -> None:
        config = TrainingConfig(
            episode_length=8,
            warmup_steps=4,
            autostart=False,
        )
        env = make_market_env(config)
        self.assertIsInstance(env, SyntheticMarketEnv)
        state = env.reset(seed=11)
        next_state, reward, done, info = env.step(2)
        self.assertEqual(state.shape, (config.observation_dim,))
        self.assertEqual(next_state.shape, (config.observation_dim,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("regime", info)

    def test_continuous_market_env_smoke(self) -> None:
        config = TrainingConfig(
            episode_length=8,
            warmup_steps=4,
            continuous_actions=True,
            autostart=False,
        )
        env = ContinuousMarketEnv(config, asset_name="spy")
        state = env.reset(seed=7)
        next_state, reward, done, info = env.step(0.35)
        self.assertEqual(state.shape, (env.observation_dim,))
        self.assertEqual(next_state.shape, (env.observation_dim,))
        self.assertEqual(len(info["positions"]), 1)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

    def test_multi_asset_continuous_env_nonstationary_smoke(self) -> None:
        config = TrainingConfig(
            episode_length=8,
            warmup_steps=4,
            continuous_actions=True,
            nonstationary_mode="cyclical",
            nonstationary_drift_scale=0.4,
            real_data_symbols=("SPY", "QQQ", "GLD"),
            autostart=False,
        )
        env = MultiAssetContinuousMarketEnv(config)
        state = env.reset(seed=13)
        next_state, reward, done, info = env.step(np.array([0.2, -0.3, 0.1], dtype=np.float32))
        self.assertEqual(state.shape, (env.observation_dim,))
        self.assertEqual(next_state.shape, (env.observation_dim,))
        self.assertEqual(len(info["positions"]), 3)
        self.assertIn("portfolio.gross_exposure", info["feature_map"])
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

    def test_actor_critic_builder_variants_smoke(self) -> None:
        base_config = TrainingConfig(hidden_dim=32, gate_hidden_dim=16, n_experts=3, autostart=False)
        observation = torch.zeros((2, 7), dtype=torch.float32)

        ppo = build_actor_critic(
            replace(base_config, algorithm=AlgorithmType.PPO),
            observation_dim=7,
            action_dim=1,
            device="cpu",
        )
        sac = build_actor_critic(
            replace(base_config, algorithm=AlgorithmType.SAC),
            observation_dim=7,
            action_dim=1,
            device="cpu",
        )
        rcmoe = build_actor_critic(
            replace(base_config, agent_type=AgentType.RCMOE_DQN),
            observation_dim=7,
            action_dim=1,
            device="cpu",
        )

        self.assertIsInstance(ppo, PPOActorCritic)
        self.assertIsInstance(sac, SACActorCritic)
        self.assertIsInstance(rcmoe, RCMoEActorCritic)
        self.assertEqual(ppo.forward(observation)["mean"].shape, (2, 1))
        self.assertEqual(sac.forward(observation)["mean"].shape, (2, 1))
        self.assertEqual(rcmoe.forward(observation)["gate_weights"].shape, (2, 3))

    def test_explainability_helpers_smoke(self) -> None:
        config = TrainingConfig(hidden_dim=32, gate_hidden_dim=16, n_experts=3, autostart=False)
        model = build_actor_critic(
            replace(config, agent_type=AgentType.RCMOE_DQN),
            observation_dim=7,
            action_dim=1,
            device="cpu",
        )
        state = np.linspace(-0.5, 0.5, 7, dtype=np.float32)

        attribution = gate_attribution(model, state, steps=8)
        counterfactual = expert_counterfactual(model, state, expert_index=1)
        boundary = decision_boundary(model, state, feature_x=0, feature_y=1, grid_size=9)
        gate_boundary = decision_boundary(model, state, feature_x=0, feature_y=1, grid_size=9, target="gate")
        lag = transition_lag(
            ["bull", "bull", "bear", "bear", "chop", "chop"],
            ["bull", "bull", "bull", "bear", "chop", "chop"],
            max_lag=3,
        )

        self.assertEqual(attribution["attribution"].shape, (7,))
        self.assertEqual(counterfactual["gate_weights"].shape, (3,))
        self.assertEqual(boundary["decision_grid"].shape, (9, 9))
        self.assertEqual(gate_boundary["target"], "gate")
        self.assertEqual(gate_boundary["decision_grid"].shape, (9, 9))
        self.assertIn("best_global_lag", lag)


if __name__ == "__main__":
    unittest.main()
