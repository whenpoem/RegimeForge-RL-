from __future__ import annotations

import unittest

import numpy as np

from backend.regime_lens.agent_io import AgentObservationAdapter
from backend.regime_lens.config import AgentType, GateType, TrainingConfig
from backend.regime_lens.market import SyntheticMarketEnv
from backend.regime_lens.rcmoe import RCMoEAgent
from backend.regime_lens.transformer_agent import TransformerDQNAgent


class AgentObservationAdapterTests(unittest.TestCase):
    def test_discrete_temporal_gate_context_is_shared_across_train_and_next_observation(self) -> None:
        config = TrainingConfig(
            agent_type=AgentType.RCMOE_DQN,
            gate_type=GateType.TEMPORAL,
            context_len=3,
            autostart=False,
            hidden_dim=16,
            gate_hidden_dim=8,
            n_experts=2,
            replay_capacity=16,
            batch_size=2,
            seed=101,
        )
        agent = RCMoEAgent(
            observation_dim=config.observation_dim * config.context_len,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            n_experts=config.n_experts,
            gate_hidden_dim=config.gate_hidden_dim,
            load_balance_weight=config.load_balance_weight,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            tau=config.tau,
            replay_capacity=config.replay_capacity,
            batch_size=config.batch_size,
            device="cpu",
            seed=config.seed,
            gate_type=config.gate_type,
            context_len=config.context_len,
        )
        env = SyntheticMarketEnv(config)
        state = env.reset(seed=101)
        adapter = AgentObservationAdapter(config, agent)
        adapter.reset_episode(env, state)

        observation = adapter.discrete_observation(env, state)
        next_state, _, _, _ = env.step(1)
        next_observation = adapter.next_discrete_observation(env, next_state)
        adapter.advance(next_state)
        current_after_advance = adapter.discrete_observation(env, next_state)

        np.testing.assert_allclose(observation, np.tile(state, config.context_len))
        np.testing.assert_allclose(next_observation, current_after_advance)
        np.testing.assert_allclose(
            next_observation,
            np.concatenate([state, state, next_state]).astype(np.float32),
        )

    def test_transformer_surface_and_explainability_observations_are_flattened_sequences(self) -> None:
        config = TrainingConfig(
            agent_type=AgentType.TRANSFORMER_DQN,
            seq_len=4,
            autostart=False,
            hidden_dim=16,
            n_heads=2,
            n_layers=1,
            replay_capacity=16,
            batch_size=2,
            seed=103,
        )
        agent = TransformerDQNAgent(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            tau=config.tau,
            replay_capacity=config.replay_capacity,
            batch_size=config.batch_size,
            device="cpu",
            seed=config.seed,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            seq_len=config.seq_len,
            dropout=0.0,
        )
        adapter = AgentObservationAdapter(config, agent)
        base_states = np.arange(2 * config.observation_dim, dtype=np.float32).reshape(2, config.observation_dim)

        policy_observations = adapter.batch_policy_observations(base_states)
        explanation_observation = adapter.explainability_observation(base_states[0])

        self.assertEqual(policy_observations.shape, (2, config.observation_dim * config.seq_len))
        self.assertEqual(explanation_observation.shape, (config.observation_dim * config.seq_len,))
        np.testing.assert_allclose(policy_observations[0], np.tile(base_states[0], config.seq_len))
        np.testing.assert_allclose(explanation_observation, np.tile(base_states[0], config.seq_len))

    def test_oracle_neutral_surface_observation_is_centralized(self) -> None:
        config = TrainingConfig(agent_type=AgentType.ORACLE_DQN, autostart=False)
        agent = TransformerDQNAgent(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_dim=16,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            tau=config.tau,
            replay_capacity=16,
            batch_size=2,
            device="cpu",
            seed=107,
            n_heads=1,
            n_layers=1,
            seq_len=2,
            dropout=0.0,
        )
        adapter = AgentObservationAdapter(config, agent)
        state = np.ones(config.observation_dim, dtype=np.float32)

        surface_state = adapter.neutral_surface_observation(state)

        self.assertEqual(surface_state.shape, (config.observation_dim + config.n_regimes,))
        self.assertAlmostEqual(float(surface_state[config.observation_dim :].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
