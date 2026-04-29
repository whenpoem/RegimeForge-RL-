from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT), str(BACKEND_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from backend.regime_lens.actor_critic import (
        PPOActorCritic,
        RCMoEActorCritic,
        RCMoESACActorCritic,
        SACActorCritic,
    )
    from backend.regime_lens.config import AgentType, AlgorithmType, GateType, TrainingConfig
    from backend.regime_lens.continuous_agent import ContinuousActorCriticAgent
except ModuleNotFoundError:
    from regime_lens.actor_critic import (
        PPOActorCritic,
        RCMoEActorCritic,
        RCMoESACActorCritic,
        SACActorCritic,
    )
    from regime_lens.config import AgentType, AlgorithmType, GateType, TrainingConfig
    from regime_lens.continuous_agent import ContinuousActorCriticAgent


class ContinuousActorCriticAgentTests(unittest.TestCase):
    def test_construction_selects_expected_backbone(self) -> None:
        ppo_agent = ContinuousActorCriticAgent(
            TrainingConfig(algorithm=AlgorithmType.PPO, device="cpu", autostart=False),
            observation_dim=5,
            action_dim=2,
        )
        self.assertIsInstance(ppo_agent.model, PPOActorCritic)

        sac_agent = ContinuousActorCriticAgent(
            TrainingConfig(algorithm=AlgorithmType.SAC, device="cpu", autostart=False),
            observation_dim=5,
            action_dim=2,
        )
        self.assertIsInstance(sac_agent.model, SACActorCritic)

        rcmoe_agent = ContinuousActorCriticAgent(
            TrainingConfig(
                algorithm=AlgorithmType.PPO,
                agent_type=AgentType.RCMOE_DQN,
                device="cpu",
                autostart=False,
                n_experts=3,
                gate_hidden_dim=12,
            ),
            observation_dim=5,
            action_dim=2,
        )
        self.assertIsInstance(rcmoe_agent.model, RCMoEActorCritic)

        rcmoe_sac_agent = ContinuousActorCriticAgent(
            TrainingConfig(
                algorithm=AlgorithmType.SAC,
                agent_type=AgentType.RCMOE_DQN,
                device="cpu",
                autostart=False,
                n_experts=3,
                gate_hidden_dim=12,
            ),
            observation_dim=5,
            action_dim=2,
        )
        self.assertIsInstance(rcmoe_sac_agent.model, RCMoESACActorCritic)
        self.assertEqual(rcmoe_sac_agent.variant, "rcmoe")

    def test_ppo_update_runs_on_synthetic_rollout(self) -> None:
        config = TrainingConfig(
            algorithm=AlgorithmType.PPO,
            device="cpu",
            autostart=False,
            batch_size=4,
            hidden_dim=32,
            seed=11,
        )
        agent = ContinuousActorCriticAgent(config, observation_dim=4, action_dim=2, ppo_epochs=2)
        baseline = {key: value.detach().clone() for key, value in agent.model.state_dict().items()}

        for step in range(8):
            observation = np.linspace(-0.3, 0.4, 4, dtype=np.float32) + step * 0.05
            action_info = agent.act(observation)
            next_observation = observation + 0.02
            agent.store(
                observation,
                action_info["action"],
                reward=0.1 * (step + 1),
                next_observation=next_observation,
                done=step == 7,
                log_prob=action_info["log_prob"],
                value=action_info["value"],
            )

        metrics = agent.update()

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertIn("policy_loss", metrics)
        self.assertEqual(len(agent.buffer), 0)
        changed = any(
            not torch.equal(value, baseline[key])
            for key, value in agent.model.state_dict().items()
        )
        self.assertTrue(changed)

    def test_sac_update_and_checkpoint_round_trip(self) -> None:
        config = TrainingConfig(
            algorithm=AlgorithmType.SAC,
            device="cpu",
            autostart=False,
            batch_size=4,
            replay_capacity=32,
            hidden_dim=32,
            gradient_steps=2,
            seed=23,
        )
        agent = ContinuousActorCriticAgent(config, observation_dim=4, action_dim=2)

        for step in range(12):
            observation = np.array(
                [step * 0.1, -0.2 + step * 0.03, 0.1 - step * 0.02, 0.05 * step],
                dtype=np.float32,
            )
            action = np.array(
                [np.sin(step * 0.2), np.cos(step * 0.3)],
                dtype=np.float32,
            ) * 0.4
            next_observation = observation + np.array([0.02, -0.01, 0.015, 0.005], dtype=np.float32)
            reward = float(np.tanh(step * 0.15))
            agent.store(observation, action, reward, next_observation, done=step % 5 == 4)

        metrics = agent.update()

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertIn("critic_loss", metrics)
        self.assertGreater(len(agent.buffer), 0)

        reference_observation = np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float32)
        expected_action = agent.act(reference_observation, deterministic=True)["action"]

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "continuous-agent"
            agent.save_checkpoint(checkpoint_dir)

            restored = ContinuousActorCriticAgent(config, observation_dim=4, action_dim=2)
            restored.load_checkpoint(checkpoint_dir, weights_only=False)

            restored_action = restored.act(reference_observation, deterministic=True)["action"]
            np.testing.assert_allclose(restored_action, expected_action, atol=1e-5)
            self.assertEqual(len(restored.buffer), len(agent.buffer))
            self.assertAlmostEqual(float(restored.alpha.item()), float(agent.alpha.item()), places=6)

            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key])
            assert agent.target_model is not None
            assert restored.target_model is not None
            for key, value in agent.target_model.state_dict().items():
                torch.testing.assert_close(value, restored.target_model.state_dict()[key])

    def test_sac_rcmoe_update_exposes_gate_metrics(self) -> None:
        config = TrainingConfig(
            algorithm=AlgorithmType.SAC,
            agent_type=AgentType.RCMOE_DQN,
            device="cpu",
            autostart=False,
            batch_size=4,
            replay_capacity=32,
            hidden_dim=24,
            gate_hidden_dim=16,
            n_experts=3,
            gradient_steps=2,
            load_balance_weight=0.05,
            seed=29,
        )
        agent = ContinuousActorCriticAgent(config, observation_dim=4, action_dim=2)

        self.assertIsInstance(agent.model, RCMoESACActorCritic)

        for step in range(12):
            observation = np.array(
                [0.05 * step, -0.1 + step * 0.02, 0.15 - step * 0.01, -0.2 + step * 0.03],
                dtype=np.float32,
            )
            action = np.array(
                [np.sin(step * 0.3), np.cos(step * 0.25)],
                dtype=np.float32,
            ) * 0.35
            next_observation = observation + np.array([0.01, -0.015, 0.02, 0.005], dtype=np.float32)
            reward = float(np.tanh(step * 0.12))
            agent.store(observation, action, reward, next_observation, done=step % 4 == 3)

        metrics = agent.update()

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertIn("critic_loss", metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("load_balance_loss", metrics)
        gate_weights = agent.gate_weights(np.array([0.2, -0.1, 0.05, 0.1], dtype=np.float32))
        self.assertIsNotNone(gate_weights)
        assert gate_weights is not None
        self.assertEqual(gate_weights.shape, (3,))
        self.assertAlmostEqual(float(gate_weights.sum()), 1.0, places=5)

    def test_temporal_rcmoe_stores_contexts_without_query_side_effects(self) -> None:
        config = TrainingConfig(
            algorithm=AlgorithmType.SAC,
            agent_type=AgentType.RCMOE_DQN,
            gate_type=GateType.TEMPORAL,
            context_len=3,
            device="cpu",
            autostart=False,
            batch_size=2,
            replay_capacity=8,
            hidden_dim=16,
            gate_hidden_dim=8,
            n_experts=2,
            seed=31,
        )
        agent = ContinuousActorCriticAgent(config, observation_dim=2, action_dim=1)
        first = np.array([0.1, -0.2], dtype=np.float32)
        second = np.array([0.3, 0.4], dtype=np.float32)

        action = agent.act(first, deterministic=True)["action"]
        context_before_query = [item.copy() for item in agent._context_history]
        gate_weights = agent.gate_weights(first)
        context_after_query = [item.copy() for item in agent._context_history]

        self.assertIsNotNone(gate_weights)
        for before, after in zip(context_before_query, context_after_query, strict=True):
            np.testing.assert_allclose(before, after)

        agent.store(first, action, reward=0.2, next_observation=second, done=False)
        assert hasattr(agent.buffer, "observations")
        np.testing.assert_allclose(agent.buffer.observations[0], np.tile(first, 3))
        np.testing.assert_allclose(agent.buffer.next_observations[0], np.concatenate([first, first, second]))


if __name__ == "__main__":
    unittest.main()
