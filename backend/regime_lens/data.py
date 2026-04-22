from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .config import DataSource, REGIME_LABELS, TrainingConfig


def inject_fitted_regime_data(config: TrainingConfig) -> tuple[TrainingConfig, dict[str, Any] | None]:
    if config.data_source == DataSource.SYNTHETIC:
        return config, None

    prices = load_price_data(config)
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Real-data pipeline requires at least two price samples.")

    portfolio_returns = returns.mean(axis=1).to_numpy(dtype=np.float64)
    fit = fit_regime_model(portfolio_returns, n_regimes=config.n_regimes)

    updated = replace(
        config,
        regime_transition=fit["regime_transition"],
        regime_params=fit["regime_params"],
    )
    return updated, {
        "data_source": config.data_source.value,
        "symbols": list(prices.columns),
        "n_samples": int(len(prices)),
        "start": str(prices.index.min()) if len(prices.index) else None,
        "end": str(prices.index.max()) if len(prices.index) else None,
        **fit,
    }


def load_price_data(config: TrainingConfig) -> pd.DataFrame:
    if config.data_cache_path is None:
        raise ValueError("data_cache_path is required when data_source is not synthetic.")
    path = config.data_cache_path.expanduser().resolve()
    if path.is_dir():
        frames = []
        for symbol in config.real_data_symbols:
            symbol_path = path / f"{symbol}.csv"
            frames.append(_read_symbol_csv(symbol_path, symbol))
        return pd.concat(frames, axis=1).dropna()
    if path.suffix.lower() == ".csv":
        return _read_multi_symbol_csv(path, config.real_data_symbols)
    raise ValueError(f"Unsupported data cache path: {path}")


def fit_regime_model(
    portfolio_returns: np.ndarray,
    *,
    n_regimes: int,
    window_size: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    features = _rolling_features(portfolio_returns, window_size=window_size)
    if len(features) < n_regimes:
        raise ValueError("Not enough samples to fit the requested number of regimes.")

    model = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        n_init=5,
        random_state=seed,
    )
    model.fit(features)
    states = model.predict(features).astype(np.int64)
    aligned_returns = portfolio_returns[window_size - 1 :]

    state_stats = _summarise_states(aligned_returns, states, n_regimes)
    label_mapping = _map_states_to_regimes(state_stats)
    regime_transition = _estimate_transition(states, label_mapping)
    regime_params = _estimate_regime_params(aligned_returns, states, label_mapping)

    return {
        "window_size": window_size,
        "label_mapping": {str(key): value for key, value in label_mapping.items()},
        "regime_transition": regime_transition,
        "regime_params": regime_params,
    }


def _read_symbol_csv(path: Path, symbol: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected cached CSV for symbol {symbol} at {path}")
    frame = pd.read_csv(path)
    close_column = _pick_close_column(frame.columns)
    date_column = _pick_date_column(frame.columns)
    out = pd.DataFrame(
        {
            symbol: pd.to_numeric(frame[close_column], errors="coerce").to_numpy(),
        },
        index=pd.to_datetime(frame[date_column], utc=True),
    )
    return out.sort_index().dropna()


def _read_multi_symbol_csv(path: Path, symbols: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    lowered = {column.lower(): column for column in frame.columns}
    date_column = _pick_date_column(frame.columns)
    symbol_column = lowered.get("symbol")
    close_column = _pick_close_column(frame.columns)
    if symbol_column is not None:
        pivot = frame.pivot_table(
            index=date_column,
            columns=symbol_column,
            values=close_column,
            aggfunc="last",
        )
        if symbols:
            pivot = pivot[[symbol for symbol in symbols if symbol in pivot.columns]]
        pivot.index = pd.to_datetime(pivot.index, utc=True)
        return pivot.sort_index().dropna()

    keep = [column for column in frame.columns if column in symbols]
    if not keep:
        raise ValueError(f"CSV file {path} does not contain any of the requested symbols {symbols}.")
    out = frame[[date_column, *keep]].copy()
    out[date_column] = pd.to_datetime(out[date_column], utc=True)
    return out.set_index(date_column).sort_index().dropna()


def _pick_close_column(columns: Any) -> str:
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in ("close", "adj_close", "adj close", "settle", "price"):
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Could not infer a close-price column from {list(columns)}")


def _pick_date_column(columns: Any) -> str:
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in ("date", "datetime", "timestamp"):
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Could not infer a date column from {list(columns)}")


def _rolling_features(returns: np.ndarray, *, window_size: int) -> np.ndarray:
    if len(returns) < window_size:
        return np.empty((0, 3), dtype=np.float64)
    rows: list[np.ndarray] = []
    for index in range(window_size, len(returns) + 1):
        window = returns[index - window_size : index]
        mean = float(np.mean(window))
        std = float(np.std(window)) + 1e-10
        skew = float(np.mean(((window - mean) / std) ** 3))
        rows.append(np.asarray([mean, std, skew], dtype=np.float64))
    return np.asarray(rows, dtype=np.float64)


def _summarise_states(returns: np.ndarray, states: np.ndarray, n_regimes: int) -> list[dict[str, float]]:
    summaries: list[dict[str, float]] = []
    previous_returns = np.concatenate([[0.0], returns[:-1]])
    for state_id in range(n_regimes):
        mask = states == state_id
        if not mask.any():
            summaries.append({"mean": 0.0, "vol": 0.0, "autocorr": 0.0})
            continue
        state_returns = returns[mask]
        autoreg = np.corrcoef(previous_returns[mask], state_returns)[0, 1] if mask.sum() > 1 else 0.0
        summaries.append(
            {
                "mean": float(np.mean(state_returns)),
                "vol": float(np.std(state_returns)),
                "autocorr": float(np.nan_to_num(autoreg)),
            }
        )
    return summaries


def _map_states_to_regimes(state_stats: list[dict[str, float]]) -> dict[int, str]:
    indices = list(range(len(state_stats)))
    bull_idx = max(indices, key=lambda idx: state_stats[idx]["mean"])
    bear_idx = min(indices, key=lambda idx: state_stats[idx]["mean"])
    remaining = [idx for idx in indices if idx not in {bull_idx, bear_idx}]
    shock_idx = max(remaining or indices, key=lambda idx: state_stats[idx]["vol"])
    chop_candidates = [idx for idx in indices if idx not in {bull_idx, bear_idx, shock_idx}]
    mapping = {
        bull_idx: "bull",
        bear_idx: "bear",
        shock_idx: "shock",
    }
    for candidate in chop_candidates:
        mapping[candidate] = "chop"
    for idx in indices:
        mapping.setdefault(idx, REGIME_LABELS[min(idx, len(REGIME_LABELS) - 1)])
    return mapping


def _estimate_transition(states: np.ndarray, mapping: dict[int, str]) -> tuple[tuple[float, ...], ...]:
    transition = np.full((len(REGIME_LABELS), len(REGIME_LABELS)), 1e-6, dtype=np.float64)
    regime_index = {label: idx for idx, label in enumerate(REGIME_LABELS)}
    for left, right in zip(states[:-1], states[1:], strict=True):
        transition[regime_index[mapping[int(left)]], regime_index[mapping[int(right)]]] += 1.0
    transition /= transition.sum(axis=1, keepdims=True)
    return tuple(tuple(float(value) for value in row) for row in transition)


def _estimate_regime_params(
    returns: np.ndarray,
    states: np.ndarray,
    mapping: dict[int, str],
) -> dict[str, dict[str, float]]:
    previous_returns = np.concatenate([[0.0], returns[:-1]])
    out: dict[str, dict[str, float]] = {}
    for regime in REGIME_LABELS:
        mask = np.array([mapping[int(state)] == regime for state in states], dtype=bool)
        if not mask.any():
            out[regime] = {
                "drift": 0.0,
                "vol": 0.01,
                "autocorr": 0.0,
                "jump_prob": 0.0,
                "jump_scale": 0.0,
            }
            continue
        regime_returns = returns[mask]
        threshold = float(np.percentile(np.abs(regime_returns), 90))
        jump_mask = np.abs(regime_returns) >= threshold
        autocorr = np.corrcoef(previous_returns[mask], regime_returns)[0, 1] if mask.sum() > 1 else 0.0
        out[regime] = {
            "drift": float(np.mean(regime_returns)),
            "vol": float(np.std(regime_returns) + 1e-6),
            "autocorr": float(np.clip(np.nan_to_num(autocorr), -0.9, 0.9)),
            "jump_prob": float(np.mean(jump_mask)),
            "jump_scale": float(np.std(regime_returns[jump_mask])) if jump_mask.any() else 0.0,
        }
    return out
