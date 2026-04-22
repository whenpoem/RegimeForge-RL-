from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from backend.regime_lens.config import AgentType, TrainingConfig
from backend.regime_lens.dqn import DQNAgent, PrioritizedReplayBuffer
from backend.regime_lens.hmm_dqn import HMMDQNAgent
from backend.regime_lens.rcmoe import RCMoEAgent
from backend.regime_lens.training import TrainingManager, _create_agent


class CheckpointRestoreTests(unittest.TestCase):
    def test_training_writes_explainability_artifact_for_rcmoe_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp)
            config = TrainingConfig(
                artifact_root=artifact_root,
                agent_type=AgentType.RCMOE_DQN,
                seed=5,
                episodes=1,
                episode_length=12,
                warmup_steps=6,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                train_after_steps=0,
                update_every_steps=1,
                gradient_steps=1,
                batch_size=4,
                replay_capacity=32,
                hidden_dim=24,
                gate_hidden_dim=16,
                n_experts=3,
                device="cpu",
                autostart=False,
            )
            manager = TrainingManager(config)
            run_id, _ = manager.store.create_run(manager._config_payload())

            manager._train_loop(run_id)

            checkpoint_id = manager.store.read_json(artifact_root / run_id / "checkpoints" / "index.json")["checkpoints"][0][
                "checkpointId"
            ]
            explainability = manager.store.checkpoint_explainability(run_id, checkpoint_id)

            self.assertIsNotNone(explainability)
            assert explainability is not None
            self.assertIn("policyBoundary", explainability)
            self.assertIn("gateAttribution", explainability)
            self.assertIn("gateBoundary", explainability)
            self.assertIn("expertCounterfactuals", explainability)
            self.assertIn("transitionLag", explainability)
            self.assertEqual(explainability["policyBoundary"]["target"], "policy")
            self.assertEqual(explainability["gateBoundary"]["target"], "gate")
            self.assertEqual(len(explainability["expertCounterfactuals"]), config.n_experts)

    def test_load_agent_uses_saved_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp)
            saved_config = TrainingConfig(
                artifact_root=artifact_root,
                agent_type=AgentType.RCMOE_DQN,
                hidden_dim=31,
                n_experts=3,
                gate_hidden_dim=13,
                batch_size=2,
                replay_capacity=8,
                device="cpu",
                autostart=False,
            )
            writer = TrainingManager(saved_config)
            run_id, _ = writer.store.create_run(writer._config_payload())
            checkpoint_id = "ckpt-0001"

            agent = _create_agent(saved_config, "cpu")
            agent.save_checkpoint(writer.store.model_weights_dir(run_id, checkpoint_id))

            current_config = TrainingConfig(
                artifact_root=artifact_root,
                agent_type=AgentType.DQN,
                hidden_dim=17,
                device="cpu",
                autostart=False,
            )
            reader = TrainingManager(current_config)
            loaded = reader.load_agent_from_checkpoint(checkpoint_id, run_id=run_id)

            self.assertIsInstance(loaded, RCMoEAgent)
            self.assertEqual(loaded.n_experts, saved_config.n_experts)
            self.assertEqual(loaded.online.gate.net[0].out_features, saved_config.gate_hidden_dim)
            self.assertEqual(loaded.online.experts[0].layers[0].out_features, saved_config.hidden_dim)

    def test_hmm_checkpoint_restores_detector_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp)
            config = TrainingConfig(
                artifact_root=artifact_root,
                agent_type=AgentType.HMM_DQN,
                batch_size=2,
                replay_capacity=8,
                device="cpu",
                autostart=False,
            )
            manager = TrainingManager(config)
            run_id, _ = manager.store.create_run(manager._config_payload())
            checkpoint_id = "ckpt-0001"

            agent = _create_agent(config, "cpu")
            self.assertIsInstance(agent, HMMDQNAgent)
            returns = np.array(
                [
                    -0.21,
                    -0.15,
                    -0.08,
                    -0.02,
                    0.03,
                    0.09,
                    0.14,
                    0.19,
                    0.12,
                    0.05,
                    -0.01,
                    -0.07,
                    -0.12,
                    -0.18,
                    -0.11,
                    -0.04,
                    0.02,
                    0.08,
                    0.13,
                    0.2,
                ],
                dtype=np.float64,
            )
            base_state = np.linspace(-0.3, 0.3, config.observation_dim, dtype=np.float32)
            agent.fit_detector(returns)
            expected_state = agent.augment_state(base_state, returns[-10:])
            agent.save_checkpoint(manager.store.model_weights_dir(run_id, checkpoint_id))

            loaded = manager.load_agent_from_checkpoint(checkpoint_id, run_id=run_id)
            self.assertIsInstance(loaded, HMMDQNAgent)
            self.assertIsNotNone(loaded.detector.model)
            restored_state = loaded.augment_state(base_state, returns[-10:])
            np.testing.assert_allclose(restored_state, expected_state)

    def test_resume_restores_optimizer_rng_and_prioritized_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "weights"
            agent = DQNAgent(
                observation_dim=3,
                action_dim=2,
                hidden_dim=8,
                learning_rate=1e-3,
                gamma=0.95,
                tau=0.1,
                replay_capacity=8,
                batch_size=2,
                device="cpu",
                seed=7,
                use_per=True,
            )
            for index in range(4):
                state = np.array([index, index + 0.5, -index], dtype=np.float32)
                next_state = state + 0.25
                agent.store(
                    state,
                    action=index % 2,
                    reward=float(index + 1) * 0.1,
                    next_state=next_state,
                    done=index == 3,
                )

            loss = agent.update(per_beta=0.7)
            self.assertIsNotNone(loss)
            agent.save_checkpoint(checkpoint_dir)

            loaded = DQNAgent(
                observation_dim=3,
                action_dim=2,
                hidden_dim=8,
                learning_rate=1e-3,
                gamma=0.95,
                tau=0.1,
                replay_capacity=8,
                batch_size=2,
                device="cpu",
                seed=7,
                use_per=True,
            )
            loaded.load_checkpoint(checkpoint_dir, weights_only=False)

            self.assertIsInstance(loaded.buffer, PrioritizedReplayBuffer)
            self.assertEqual(loaded.buffer.size, agent.buffer.size)
            self.assertEqual(loaded.buffer.index, agent.buffer.index)
            np.testing.assert_allclose(loaded.buffer.states[: agent.buffer.size], agent.buffer.states[: agent.buffer.size])
            np.testing.assert_allclose(
                loaded.buffer.next_states[: agent.buffer.size],
                agent.buffer.next_states[: agent.buffer.size],
            )
            np.testing.assert_array_equal(loaded.buffer.actions[: agent.buffer.size], agent.buffer.actions[: agent.buffer.size])
            np.testing.assert_allclose(loaded.buffer.rewards[: agent.buffer.size], agent.buffer.rewards[: agent.buffer.size])
            np.testing.assert_allclose(loaded.buffer.dones[: agent.buffer.size], agent.buffer.dones[: agent.buffer.size])
            np.testing.assert_allclose(loaded.buffer.tree.tree, agent.buffer.tree.tree)
            self.assertEqual(loaded.buffer.tree.write_index, agent.buffer.tree.write_index)
            self.assertAlmostEqual(loaded.buffer._max_priority, agent.buffer._max_priority)

            left = agent.buffer.sample(2, np.random.default_rng(1234), beta=0.85)
            right = loaded.buffer.sample(2, np.random.default_rng(1234), beta=0.85)
            for original, restored in zip(left, right, strict=True):
                np.testing.assert_allclose(original, restored)

            self.assertEqual(int(agent.rng.integers(0, 100_000)), int(loaded.rng.integers(0, 100_000)))

            original_optimizer = agent.optimizer.state_dict()["state"]
            restored_optimizer = loaded.optimizer.state_dict()["state"]
            self.assertEqual(set(original_optimizer), set(restored_optimizer))
            self.assertTrue(restored_optimizer)
            for param_id, state in original_optimizer.items():
                for key, value in state.items():
                    restored_value = restored_optimizer[param_id][key]
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, restored_value)
                    else:
                        self.assertEqual(value, restored_value)


if __name__ == "__main__":
    unittest.main()
