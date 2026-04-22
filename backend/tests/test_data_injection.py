from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.regime_lens.config import DataSource, REGIME_LABELS, TrainingConfig
from backend.regime_lens.data import inject_fitted_regime_data


class DataInjectionTests(unittest.TestCase):
    def test_inject_fitted_regime_data_updates_config_from_csv_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()

            dates = pd.date_range("2024-01-01", periods=64, freq="D", tz="UTC")
            base_returns = np.array(
                ([0.012] * 16)
                + ([-0.01] * 16)
                + ([0.003, -0.002] * 8)
                + ([0.028, -0.024, 0.022, -0.02] * 4),
                dtype=np.float64,
            )
            offsets = {
                "SPY": 0.0000,
                "QQQ": 0.0015,
                "GLD": -0.0010,
            }

            for symbol, offset in offsets.items():
                symbol_returns = np.clip(base_returns + offset, -0.2, 0.2)
                prices = 100.0 * np.cumprod(1.0 + symbol_returns)
                frame = pd.DataFrame({"Date": dates, "Close": prices})
                frame.to_csv(cache_dir / f"{symbol}.csv", index=False)

            original = TrainingConfig(
                artifact_root=root / "artifacts",
                data_source=DataSource.CSV,
                data_cache_path=cache_dir,
                real_data_symbols=tuple(offsets),
                autostart=False,
            )

            updated, data_fit = inject_fitted_regime_data(original)

            self.assertIsNotNone(data_fit)
            assert data_fit is not None
            self.assertEqual(data_fit["data_source"], DataSource.CSV.value)
            self.assertEqual(data_fit["symbols"], list(offsets))
            self.assertEqual(data_fit["n_samples"], len(dates))
            self.assertEqual(data_fit["window_size"], 10)
            self.assertEqual(set(data_fit["label_mapping"].values()), set(REGIME_LABELS))

            self.assertEqual(len(updated.regime_transition), len(REGIME_LABELS))
            for row in updated.regime_transition:
                self.assertEqual(len(row), len(REGIME_LABELS))
                self.assertAlmostEqual(sum(row), 1.0, places=6)

            self.assertEqual(set(updated.regime_params), set(REGIME_LABELS))
            for regime in REGIME_LABELS:
                params = updated.regime_params[regime]
                self.assertIn("drift", params)
                self.assertIn("vol", params)
                self.assertIn("autocorr", params)
                self.assertIn("jump_prob", params)
                self.assertIn("jump_scale", params)

            self.assertNotEqual(updated.regime_transition, original.regime_transition)
            self.assertNotEqual(updated.regime_params, original.regime_params)


if __name__ == "__main__":
    unittest.main()
