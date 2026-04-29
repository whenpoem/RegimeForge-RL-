from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.regime_lens.config import AgentType, AlgorithmType, TrainingConfig
from backend.regime_lens.run_experiments import run_experiments
from backend.regime_lens.training import TrainingManager


class ContinuousTrainingTests(unittest.TestCase):
    def test_training_manager_continuous_ppo_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            config = TrainingConfig(
                artifact_root=artifact_root,
                algorithm=AlgorithmType.PPO,
                continuous_actions=True,
                real_data_symbols=("SPY",),
                device="cpu",
                autostart=False,
                episodes=1,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                batch_size=4,
                hidden_dim=32,
                seed=41,
            )
            manager = TrainingManager(config)
            run_id, _ = manager.store.create_run(manager._config_payload())

            manager._train_loop(run_id)

            checkpoint_index = manager.checkpoints(run_id)
            self.assertIsNotNone(checkpoint_index)
            assert checkpoint_index is not None
            self.assertEqual(len(checkpoint_index["checkpoints"]), 1)
            checkpoint_id = checkpoint_index["checkpoints"][0]["checkpointId"]
            policy = manager.checkpoint_policy(checkpoint_id, run_id=run_id)
            stats = manager.checkpoint_stats(checkpoint_id, run_id=run_id)

            self.assertIsNotNone(policy)
            self.assertIsNotNone(stats)
            assert policy is not None
            self.assertIn("cells", policy)
            self.assertIn("agentReturnMean", stats)

    def test_run_experiments_continuous_sac_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            config = TrainingConfig(
                artifact_root=artifact_root,
                experiment_name="continuous_smoke",
                algorithm=AlgorithmType.SAC,
                continuous_actions=True,
                real_data_symbols=("SPY",),
                device="cpu",
                autostart=False,
                episodes=1,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                batch_size=4,
                replay_capacity=32,
                gradient_steps=1,
                hidden_dim=32,
                seed=43,
                seeds=(43,),
                fixed_eval_seeds=(9001,),
            )

            result = run_experiments(
                "smoke",
                config,
                output_root=artifact_root / "_experiments" / config.experiment_name,
            )

            self.assertEqual(result["suite"], "smoke")
            self.assertEqual(len(result["executionRecords"]), 3)
            self.assertTrue(all(record["status"] == "completed" for record in result["executionRecords"]))
            trained = [record for record in result["executionRecords"] if record["checkpointId"] is not None]
            self.assertGreaterEqual(len(trained), 2)
            self.assertTrue(any(record["result"]["seedResult"]["extra"].get("explainability") is not None for record in trained))

    def test_run_experiments_continuous_sac_rcmoe_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            config = TrainingConfig(
                artifact_root=artifact_root,
                experiment_name="continuous_sac_rcmoe_smoke",
                algorithm=AlgorithmType.SAC,
                agent_type=AgentType.RCMOE_DQN,
                continuous_actions=True,
                real_data_symbols=("SPY",),
                device="cpu",
                autostart=False,
                episodes=1,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                batch_size=4,
                replay_capacity=32,
                gradient_steps=1,
                hidden_dim=32,
                gate_hidden_dim=16,
                n_experts=3,
                load_balance_weight=0.05,
                seed=45,
                seeds=(45,),
                fixed_eval_seeds=(9001,),
            )

            result = run_experiments(
                "smoke",
                config,
                output_root=artifact_root / "_experiments" / config.experiment_name,
            )

            trained = [record for record in result["executionRecords"] if record["checkpointId"] is not None]
            rcmoe_record = next(record for record in trained if record["specKey"] == "smoke_rcmoe")

            self.assertEqual(rcmoe_record["status"], "completed")
            self.assertEqual(rcmoe_record["result"]["seedResult"]["extra"]["agentType"], "rcmoe_dqn")
            self.assertIsNotNone(rcmoe_record["result"]["seedResult"]["extra"].get("explainability"))
            self.assertGreater(
                rcmoe_record["result"]["seedResult"]["extra"]["stats"]["agentReturnMean"],
                float("-inf"),
            )

    def test_run_experiments_continuous_ppo_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            config = TrainingConfig(
                artifact_root=artifact_root,
                experiment_name="continuous_ppo_smoke",
                algorithm=AlgorithmType.PPO,
                continuous_actions=True,
                real_data_symbols=("SPY",),
                device="cpu",
                autostart=False,
                episodes=1,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                batch_size=4,
                hidden_dim=32,
                gate_hidden_dim=16,
                n_experts=3,
                seed=47,
                seeds=(47,),
                fixed_eval_seeds=(9001,),
            )

            result = run_experiments(
                "smoke",
                config,
                output_root=artifact_root / "_experiments" / config.experiment_name,
            )

            self.assertEqual(len(result["executionRecords"]), 3)
            self.assertTrue(all(record["status"] == "completed" for record in result["executionRecords"]))
            trained = [record for record in result["executionRecords"] if record["checkpointId"] is not None]
            self.assertEqual(len(trained), 2)
            self.assertTrue(all(record["result"]["seedResult"]["extra"].get("explainability") is not None for record in trained))

    def test_multi_asset_nonstationary_continuous_sac_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            config = TrainingConfig(
                artifact_root=artifact_root,
                experiment_name="continuous_multi_asset_smoke",
                algorithm=AlgorithmType.SAC,
                continuous_actions=True,
                real_data_symbols=("SPY", "QQQ", "GLD"),
                nonstationary_mode="cyclical",
                nonstationary_drift_scale=0.35,
                device="cpu",
                autostart=False,
                episodes=1,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                batch_size=4,
                replay_capacity=32,
                gradient_steps=1,
                hidden_dim=32,
                seed=53,
                seeds=(53,),
                fixed_eval_seeds=(9001,),
            )

            manager = TrainingManager(config)
            run_id, _ = manager.store.create_run(manager._config_payload())
            manager._train_loop(run_id)

            checkpoint_index = manager.checkpoints(run_id)
            self.assertIsNotNone(checkpoint_index)
            assert checkpoint_index is not None
            checkpoint_id = checkpoint_index["checkpoints"][0]["checkpointId"]
            summary = manager.latest_run_summary() if manager.current_run_id == run_id else manager.store.read_run_summary(run_id)
            policy = manager.checkpoint_policy(checkpoint_id, run_id=run_id)

            self.assertEqual(summary["actionLabels"], ["SPY", "QQQ", "GLD"])
            assert policy is not None
            first_action = policy["cells"][0]["action"]
            self.assertEqual(len(first_action), 3)

    def test_ood_drift_suite_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            config = TrainingConfig(
                artifact_root=artifact_root,
                experiment_name="continuous_ood_drift",
                algorithm=AlgorithmType.SAC,
                continuous_actions=True,
                real_data_symbols=("SPY", "QQQ", "GLD"),
                device="cpu",
                autostart=False,
                episodes=1,
                episode_length=6,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                batch_size=4,
                replay_capacity=32,
                gradient_steps=1,
                hidden_dim=24,
                seed=59,
                seeds=(59,),
                fixed_eval_seeds=(9001,),
            )

            result = run_experiments(
                "ood",
                config,
                output_root=artifact_root / "_experiments" / config.experiment_name,
                ood_kind="drift",
            )

            self.assertEqual(result["suite"], "ood-drift")
            self.assertEqual(len(result["executionRecords"]), 8)
            self.assertTrue(all(record["status"] == "completed" for record in result["executionRecords"]))


if __name__ == "__main__":
    unittest.main()
