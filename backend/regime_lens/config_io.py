from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

import yaml

from .config import TrainingConfig, config_from_snapshot, config_to_snapshot


def load_config_payload(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = _read_payload(resolved)
    inherited = payload.pop("base", None)
    if inherited is None:
        return payload

    base_paths = inherited if isinstance(inherited, list) else [inherited]
    merged: dict[str, Any] = {}
    for candidate in base_paths:
        base_path = (resolved.parent / str(candidate)).resolve()
        merged = _deep_merge(merged, load_config_payload(base_path))
    return _deep_merge(merged, payload)


def load_training_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainingConfig:
    resolved = Path(path).expanduser().resolve()
    payload = load_config_payload(resolved)
    if overrides:
        payload = _deep_merge(payload, overrides)
    if payload.get("config_path") is None:
        payload["config_path"] = str(resolved)
    return config_from_snapshot(payload)


def dump_config_snapshot(config: TrainingConfig, path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    snapshot = config_to_snapshot(config)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        resolved.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True), encoding="utf-8")
    elif suffix == ".toml":
        raise ValueError("TOML export is not implemented; use JSON or YAML for snapshots.")
    else:
        resolved.write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return resolved


def _read_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration file {path} did not resolve to a mapping.")
        return payload
    raise ValueError(f"Unsupported config format for {path}")


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
