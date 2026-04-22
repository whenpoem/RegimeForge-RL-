from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.regime_lens.web import create_app, _encode_path_id


class WebAppTests(unittest.TestCase):
    def test_routes_and_websocket_surface_full_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "_experiments" / "suite" / "_worker_runs" / "method" / "seed-1" / "run-123"
            checkpoint_dir = run_dir / "checkpoints" / "ckpt-0001"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            summary = {
                "runId": "run-123",
                "status": "completed",
                "agentType": "rcmoe_dqn",
                "currentEpisode": 1,
                "latestCheckpointId": "ckpt-0001",
                "updatedAt": "2026-04-22T00:00:00Z",
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "metrics.json").write_text(
                json.dumps({"series": [{"episode": 1, "totalReward": 1.25}]}),
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "index.json").write_text(
                json.dumps({"checkpoints": [{"checkpointId": "ckpt-0001", "episode": 1, "agentReturn": 0.42}]}),
                encoding="utf-8",
            )
            for name, payload in {
                "summary.json": {"checkpointId": "ckpt-0001"},
                "episode.json": {"trace": []},
                "policy.json": {"cells": []},
                "embedding.json": {"points": []},
                "regime_analysis.json": {"nmi": 0.1},
                "expert_analysis.json": {"expert_utilization": [0.5, 0.5]},
                "stats.json": {"agentReturnMean": 0.42},
                "resume_state.json": {"episode": 1},
                "data_fit.json": {"symbols": ["SPY"]},
                "explainability.json": {"policyBoundary": {"target": "policy"}},
                "repro.json": {"tracking": {"actualBackend": "none"}},
            }.items():
                (checkpoint_dir / name).write_text(json.dumps(payload), encoding="utf-8")

            locator = _encode_path_id("_experiments/suite/_worker_runs/method/seed-1/run-123")
            client = TestClient(create_app(root))

            runs_json = client.get("/runs", headers={"accept": "application/json"})
            self.assertEqual(runs_json.status_code, 200)
            self.assertEqual(runs_json.json()[0]["pathId"], locator)

            runs_html = client.get("/runs", headers={"accept": "text/html"})
            self.assertEqual(runs_html.status_code, 200)
            self.assertIn("Regime Lens Dashboard", runs_html.text)

            run_detail = client.get(f"/runs/{locator}", headers={"accept": "application/json"})
            self.assertEqual(run_detail.status_code, 200)
            self.assertEqual(run_detail.json()["summary"]["runId"], "run-123")

            checkpoint_detail = client.get(
                f"/checkpoints/{locator}/ckpt-0001",
                headers={"accept": "application/json"},
            )
            self.assertEqual(checkpoint_detail.status_code, 200)
            checkpoint_payload = checkpoint_detail.json()
            self.assertIn("regime_analysis", checkpoint_payload)
            self.assertIn("expert_analysis", checkpoint_payload)
            self.assertIn("repro", checkpoint_payload)
            self.assertEqual(checkpoint_payload["explainability"]["policyBoundary"]["target"], "policy")

            with client.websocket_connect(f"/live/{locator}") as websocket:
                payload = websocket.receive_json()
                self.assertEqual(payload["runId"], "run-123")
                self.assertEqual(payload["checkpointCount"], 1)
                self.assertEqual(payload["latestCheckpoint"]["checkpointId"], "ckpt-0001")


if __name__ == "__main__":
    unittest.main()
