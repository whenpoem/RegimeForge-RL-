from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.regime_lens.config import AlgorithmType, TrackingBackend, TrainingConfig
from backend.regime_lens.tracking import NullTracker, create_tracker
from backend.regime_lens.training import TrainingManager


class TrackingTests(unittest.TestCase):
    def test_create_tracker_reports_tensorboard_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = TrainingConfig(
                artifact_root=Path(tmp) / "artifacts",
                tracking_backend=TrackingBackend.TENSORBOARD,
                autostart=False,
            )
            with patch("backend.regime_lens.tracking.TensorBoardTracker", side_effect=RuntimeError("missing tensorboard")):
                tracker = create_tracker(config, Path(tmp) / "run")

            self.assertIsInstance(tracker, NullTracker)
            payload = tracker.status_payload()
            self.assertTrue(payload["fallback"])
            self.assertEqual(payload["requestedBackend"], TrackingBackend.TENSORBOARD.value)
            self.assertEqual(payload["actualBackend"], TrackingBackend.NONE.value)
            self.assertIn("missing tensorboard", payload["warning"])

    def test_training_persists_tracking_fallback_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            config = TrainingConfig(
                artifact_root=artifact_root,
                algorithm=AlgorithmType.PPO,
                continuous_actions=True,
                real_data_symbols=("SPY",),
                tracking_backend=TrackingBackend.TENSORBOARD,
                device="cpu",
                autostart=False,
                episodes=1,
                episode_length=8,
                warmup_steps=4,
                checkpoint_interval=1,
                metrics_flush_interval=1,
                evaluation_episodes=1,
                batch_size=4,
                seed=67,
            )
            manager = TrainingManager(config)
            run_id, _ = manager.store.create_run(manager._config_payload())

            with patch("backend.regime_lens.tracking.TensorBoardTracker", side_effect=RuntimeError("writer unavailable")):
                manager._train_loop(run_id)

            summary = manager.store.read_run_summary(run_id)
            checkpoint_id = manager.checkpoints(run_id)["checkpoints"][0]["checkpointId"]
            checkpoint_root = artifact_root / run_id / "checkpoints" / checkpoint_id
            repro = manager.store.read_json(checkpoint_root / "repro.json")

            self.assertEqual(summary["tracking"]["requestedBackend"], TrackingBackend.TENSORBOARD.value)
            self.assertEqual(summary["tracking"]["actualBackend"], TrackingBackend.NONE.value)
            self.assertTrue(summary["tracking"]["fallback"])
            self.assertIn("writer unavailable", summary["tracking"]["warning"])
            self.assertEqual(repro["tracking"]["requestedBackend"], TrackingBackend.TENSORBOARD.value)
            self.assertTrue(repro["tracking"]["fallback"])


if __name__ == "__main__":
    unittest.main()
