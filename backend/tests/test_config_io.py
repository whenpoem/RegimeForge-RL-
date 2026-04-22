from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from backend.regime_lens.config import (
    AgentType,
    AlgorithmType,
    CurriculumMode,
    DataSource,
    GateType,
    TrackingBackend,
    TrainingConfig,
    config_to_snapshot,
)
from backend.regime_lens.config_io import dump_config_snapshot, load_config_payload, load_training_config


class ConfigIOTests(unittest.TestCase):
    def test_load_training_config_supports_inheritance_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_root = root / "artifacts"
            cache_path = root / "prices"
            cache_path.mkdir()

            base_json = root / "base.json"
            base_json.write_text(
                json.dumps(
                    {
                        "agent_type": "rcmoe_dqn",
                        "algorithm": "ppo",
                        "artifact_root": str(artifact_root),
                        "data_source": "csv",
                        "data_cache_path": str(cache_path),
                        "real_data_symbols": ["SPY", "QQQ"],
                        "tracking_backend": "none",
                        "gate_type": "mlp",
                        "context_len": 3,
                        "seeds": [7, 11],
                        "regime_params": {
                            "bull": {"drift": 0.01, "vol": 0.02, "autocorr": 0.3, "jump_prob": 0.0, "jump_scale": 0.0},
                            "bear": {"drift": -0.02, "vol": 0.03, "autocorr": 0.1, "jump_prob": 0.0, "jump_scale": 0.0},
                            "chop": {"drift": 0.0, "vol": 0.01, "autocorr": -0.2, "jump_prob": 0.0, "jump_scale": 0.0},
                            "shock": {"drift": -0.01, "vol": 0.05, "autocorr": -0.4, "jump_prob": 0.2, "jump_scale": 0.04},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            extra_toml = root / "extra.toml"
            extra_toml.write_text(
                '\n'.join(
                    [
                        'curriculum_mode = "uniform"',
                        'hidden_dim = 96',
                        'checkpoint_interval = 25',
                        'fixed_eval_seeds = [1001, 1003]',
                    ]
                ),
                encoding="utf-8",
            )

            child_yaml = root / "child.yaml"
            child_yaml.write_text(
                yaml.safe_dump(
                    {
                        "base": ["base.json", "extra.toml"],
                        "gate_type": "temporal",
                        "context_len": 5,
                        "regime_params": {
                            "bull": {"vol": 0.025},
                            "shock": {"jump_prob": 0.35},
                        },
                    },
                    sort_keys=False,
                    allow_unicode=False,
                ),
                encoding="utf-8",
            )

            payload = load_config_payload(child_yaml)
            self.assertEqual(payload["hidden_dim"], 96)
            self.assertEqual(payload["context_len"], 5)
            self.assertEqual(payload["regime_params"]["bull"]["drift"], 0.01)
            self.assertEqual(payload["regime_params"]["bull"]["vol"], 0.025)
            self.assertEqual(payload["regime_params"]["shock"]["jump_prob"], 0.35)

            config = load_training_config(child_yaml, overrides={"batch_size": 32, "tracking_backend": "tensorboard"})

            self.assertEqual(config.agent_type, AgentType.RCMOE_DQN)
            self.assertEqual(config.algorithm, AlgorithmType.PPO)
            self.assertEqual(config.curriculum_mode, CurriculumMode.UNIFORM)
            self.assertEqual(config.data_source, DataSource.CSV)
            self.assertEqual(config.gate_type, GateType.TEMPORAL)
            self.assertEqual(config.tracking_backend, TrackingBackend.TENSORBOARD)
            self.assertEqual(config.context_len, 5)
            self.assertEqual(config.batch_size, 32)
            self.assertEqual(config.seeds, (7, 11))
            self.assertEqual(config.fixed_eval_seeds, (1001, 1003))
            self.assertEqual(config.real_data_symbols, ("SPY", "QQQ"))
            self.assertEqual(config.artifact_root, artifact_root.resolve())
            self.assertEqual(config.data_cache_path, cache_path.resolve())
            self.assertEqual(config.config_path, child_yaml.resolve())
            self.assertEqual(config.regime_params["bull"]["drift"], 0.01)
            self.assertEqual(config.regime_params["bull"]["vol"], 0.025)
            self.assertEqual(config.regime_params["shock"]["jump_prob"], 0.35)

    def test_dump_config_snapshot_round_trips_yaml_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TrainingConfig(
                experiment_name="roundtrip",
                agent_type=AgentType.RCMOE_DQN,
                curriculum_mode=CurriculumMode.ADVERSARIAL,
                algorithm=AlgorithmType.SAC,
                artifact_root=root / "artifacts",
                data_source=DataSource.CSV,
                data_cache_path=root / "cache" / "prices.csv",
                real_data_symbols=("SPY", "TLT", "GLD"),
                seeds=(1, 2, 3),
                fixed_eval_seeds=(101, 103),
                gate_type=GateType.TEMPORAL,
                context_len=6,
                tracking_backend=TrackingBackend.NONE,
                autostart=False,
            )

            yaml_path = root / "snapshot.yaml"
            json_path = root / "snapshot.json"
            dump_config_snapshot(config, yaml_path)
            dump_config_snapshot(config, json_path)

            loaded_yaml = load_training_config(yaml_path)
            loaded_json = load_training_config(json_path)

            expected_yaml = replace(config, config_path=yaml_path.resolve())
            expected_json = replace(config, config_path=json_path.resolve())

            self.assertEqual(config_to_snapshot(loaded_yaml), config_to_snapshot(expected_yaml))
            self.assertEqual(config_to_snapshot(loaded_json), config_to_snapshot(expected_json))


if __name__ == "__main__":
    unittest.main()
