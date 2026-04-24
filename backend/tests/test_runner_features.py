from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.regime_lens.config import TrainingConfig
from backend.regime_lens.run_experiments import (
    _build_jobs,
    _process_pool_context,
    _run_trained_seed,
    build_suite,
    report_experiments,
    run_experiments,
)
from backend.regime_lens.training import TrainingManager


class RunnerFeatureTests(unittest.TestCase):
    def test_parallel_executor_uses_spawn_context(self) -> None:
        self.assertEqual(_process_pool_context().get_start_method(), "spawn")

    def test_resume_staging_across_artifact_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source_artifacts"
            target_root = root / "target_artifacts"

            source_config = TrainingConfig(
                artifact_root=source_root,
                autostart=False,
                episodes=1,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                train_after_steps=0,
                update_every_steps=1,
                gradient_steps=1,
                batch_size=4,
                replay_capacity=32,
                device="cpu",
                seed=71,
            )
            source_manager = TrainingManager(source_config)
            source_run_id, _ = source_manager.store.create_run(source_manager._config_payload())
            source_manager._train_loop(source_run_id)
            source_checkpoint_id = source_manager.checkpoints(source_run_id)["checkpoints"][0]["checkpointId"]

            resumed_config = TrainingConfig(
                artifact_root=target_root,
                autostart=False,
                episodes=2,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                train_after_steps=0,
                update_every_steps=1,
                gradient_steps=1,
                batch_size=4,
                replay_capacity=32,
                device="cpu",
                seed=71,
                resume_run_id=source_run_id,
                resume_checkpoint_id=source_checkpoint_id,
            )

            run_id, seed_result = _run_trained_seed(resumed_config, seed=71, source_artifact_root=source_root)

            self.assertIsNotNone(run_id)
            assert run_id is not None
            self.assertEqual(seed_result.extra["status"], "completed")
            summary = TrainingManager(resumed_config).store.read_run_summary(run_id)
            self.assertEqual(summary["resumedFromRunId"], source_run_id)
            self.assertEqual(summary["resumedFromCheckpointId"], source_checkpoint_id)
            self.assertEqual(summary["config"]["resume_run_id"], source_run_id)

        # staged copy should not replace original id in persisted config

    def test_parallel_job_roots_and_report_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            base_config = TrainingConfig(
                artifact_root=artifact_root,
                experiment_name="runner_features",
                autostart=False,
                episodes=1,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                batch_size=4,
                replay_capacity=32,
                device="cpu",
                seeds=(73,),
                fixed_eval_seeds=(9001,),
                parallel_workers=2,
            )

            suite = build_suite("smoke", base_config)
            jobs = _build_jobs(suite, artifact_root / "_experiments" / base_config.experiment_name / "plan", dry_run=False)
            trained_roots = [Path(job["artifactRoot"]) for job in jobs if job["spec"]["kind"] == "trained"]

            self.assertTrue(trained_roots)
            self.assertTrue(all("_worker_runs" in str(path) for path in trained_roots))

            executed = run_experiments(
                "smoke",
                base_config,
                output_root=artifact_root / "_experiments" / base_config.experiment_name,
            )
            rebuilt = report_experiments(Path(executed["outputs"]["manifestPath"]))

            self.assertEqual(rebuilt["suite"], "smoke")
            self.assertIn("statistics", rebuilt)
            self.assertIn("outputs", rebuilt)
            self.assertTrue(Path(rebuilt["outputs"]["reportPath"]).exists())


if __name__ == "__main__":
    unittest.main()
