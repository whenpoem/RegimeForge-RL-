from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.regime_lens.web import _list_runs, _live_payload, _resolve_run_dir


class WebDashboardTests(unittest.TestCase):
    def test_nested_worker_runs_get_unique_path_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_run = root / "_experiments" / "suite" / "_worker_runs" / "method_a" / "seed-1" / "run-duplicate"
            second_run = root / "_experiments" / "suite" / "_worker_runs" / "method_b" / "seed-1" / "run-duplicate"

            for index, run_dir in enumerate((first_run, second_run), start=1):
                (run_dir / "checkpoints" / "ckpt-0001").mkdir(parents=True, exist_ok=True)
                (run_dir / "summary.json").write_text(
                    (
                        "{\n"
                        '  "runId": "run-duplicate",\n'
                        f'  "updatedAt": "2026-04-22T00:00:0{index}Z",\n'
                        '  "status": "completed",\n'
                        f'  "agentType": "agent-{index}",\n'
                        '  "latestCheckpointId": "ckpt-0001"\n'
                        "}\n"
                    ),
                    encoding="utf-8",
                )
                (run_dir / "metrics.json").write_text('{"series": []}', encoding="utf-8")
                (run_dir / "checkpoints" / "index.json").write_text(
                    '{"checkpoints": [{"checkpointId": "ckpt-0001"}]}',
                    encoding="utf-8",
                )
                (run_dir / "checkpoints" / "ckpt-0001" / "summary.json").write_text(
                    '{"checkpointId": "ckpt-0001"}',
                    encoding="utf-8",
                )

            runs = _list_runs(root)

            self.assertEqual(len(runs), 2)
            self.assertNotEqual(runs[0]["pathId"], runs[1]["pathId"])
            self.assertIn("_experiments/suite/_worker_runs/method_a/seed-1/run-duplicate", {run["runPath"] for run in runs})
            self.assertIn("_experiments/suite/_worker_runs/method_b/seed-1/run-duplicate", {run["runPath"] for run in runs})
            self.assertIsNone(_resolve_run_dir(root, "run-duplicate"))

            resolved = _resolve_run_dir(root, runs[0]["pathId"])
            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertEqual(_live_payload(root, runs[0]["pathId"])["pathId"], runs[0]["pathId"])
            self.assertEqual(resolved.name, "run-duplicate")


if __name__ == "__main__":
    unittest.main()
