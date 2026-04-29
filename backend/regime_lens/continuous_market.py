from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import FEATURE_NAMES, REGIME_LABELS, TrainingConfig
from .market import ObservationView, SyntheticMarketEnv

_ORACLE_FEATURES = len(REGIME_LABELS)


@dataclass(slots=True)
class BoxSpace:
    low: float
    high: float
    shape: tuple[int, ...]


class _BaseContinuousMarketEnv:
    def __init__(self, config: TrainingConfig, asset_names: Sequence[str]):
        self.config = config
        self.asset_names = tuple(asset_names)
        self.num_assets = len(self.asset_names)
        self.total_steps = config.episode_length + config.warmup_steps + 1
        self._transition_matrix = np.asarray(config.regime_transition, dtype=np.float64)
        self._regime_index = {label: index for index, label in enumerate(REGIME_LABELS)}
        self.rng = np.random.default_rng(config.seed)

        self.prices = np.empty((0, self.num_assets), dtype=np.float64)
        self.returns = np.empty((0, self.num_assets), dtype=np.float64)
        self.regimes: list[str] = []
        self.t = config.warmup_steps
        self.end_index = self.t + config.episode_length
        self.positions = np.zeros(self.num_assets, dtype=np.float64)
        self.entry_prices = np.full(self.num_assets, np.nan, dtype=np.float64)
        self._regime_schedule: list[str] | None = None
        self.action_space = BoxSpace(low=-1.0, high=1.0, shape=(self.num_assets,))
        self.observation_space = BoxSpace(
            low=-4.0,
            high=4.0,
            shape=(self.num_assets * len(FEATURE_NAMES),),
        )

    @property
    def observation_dim(self) -> int:
        return self.observation_space.shape[0]

    @property
    def action_dim(self) -> int:
        return self.action_space.shape[0]

    def set_regime_schedule(self, schedule: list[str] | None) -> None:
        self._regime_schedule = schedule

    def regime_at(self, t: int) -> str:
        if 0 <= t < len(self.regimes):
            return self.regimes[t]
        return "unknown"

    def regime_index_at(self, t: int) -> int:
        return self._regime_index.get(self.regime_at(t), -1)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        if self._regime_schedule is not None and len(self._regime_schedule) >= self.total_steps:
            self.regimes = list(self._regime_schedule[: self.total_steps])
        else:
            self.regimes = self._generate_regimes(self.total_steps)

        self.returns = self._generate_returns(self.regimes)
        cumulative = np.cumsum(self.returns, axis=0)
        self.prices = 100.0 * np.exp(np.vstack([np.zeros((1, self.num_assets)), cumulative]))
        self.t = self.config.warmup_steps
        self.end_index = self.t + self.config.episode_length
        self.positions = np.zeros(self.num_assets, dtype=np.float64)
        self.entry_prices = np.full(self.num_assets, np.nan, dtype=np.float64)
        return self.observe().state

    def observe(self) -> ObservationView:
        idx = self.t
        state_parts: list[float] = []
        feature_map: dict[str, float] = {}

        for asset_idx, asset_name in enumerate(self.asset_names):
            current_price = float(self.prices[idx, asset_idx])
            recent_returns = self.returns[:idx, asset_idx]
            short_return = float(np.mean(recent_returns[max(0, idx - 3) : idx]) * 100.0)
            medium_return = float(np.mean(recent_returns[max(0, idx - 12) : idx]) * 100.0)
            volatility = float(np.std(recent_returns[max(0, idx - 10) : idx]) * 100.0)
            short_ma = float(np.mean(self.prices[max(0, idx - 5) : idx + 1, asset_idx]))
            long_ma = float(np.mean(self.prices[max(0, idx - 20) : idx + 1, asset_idx]))
            trend_gap = ((short_ma - long_ma) / current_price) * 100.0
            running_peak = float(np.max(self.prices[: idx + 1, asset_idx]))
            drawdown = ((current_price / running_peak) - 1.0) * 100.0
            unrealized = 0.0

            if not np.isnan(self.entry_prices[asset_idx]) and self.positions[asset_idx] != 0.0:
                unrealized = self.positions[asset_idx] * (
                    (current_price / float(self.entry_prices[asset_idx])) - 1.0
                ) * 100.0

            raw_features = {
                "short_return_pct": short_return,
                "medium_return_pct": medium_return,
                "volatility_pct": volatility,
                "trend_gap_pct": float(trend_gap),
                "drawdown_pct": float(drawdown),
                "position": float(self.positions[asset_idx]),
                "unrealized_pnl_pct": float(unrealized),
            }

            for feature_name, value in raw_features.items():
                feature_map[f"{asset_name}.{feature_name}"] = float(value)

            state_parts.extend(
                [
                    np.clip(short_return / 0.45, -4.0, 4.0),
                    np.clip(medium_return / 0.9, -4.0, 4.0),
                    np.clip(volatility / 1.7, -4.0, 4.0),
                    np.clip(trend_gap / 1.25, -4.0, 4.0),
                    np.clip(drawdown / 4.5, -4.0, 4.0),
                    np.clip(self.positions[asset_idx], -1.0, 1.0),
                    np.clip(unrealized / 2.2, -4.0, 4.0),
                ]
            )

        feature_map["portfolio.gross_exposure"] = float(np.abs(self.positions).sum())
        feature_map["portfolio.net_exposure"] = float(self.positions.sum())
        return ObservationView(state=np.asarray(state_parts, dtype=np.float32), feature_map=feature_map)

    def observe_oracle(self) -> ObservationView:
        """Return observation with regime one-hot appended (for Oracle agents)."""
        base = self.observe()
        regime_onehot = np.zeros(len(REGIME_LABELS), dtype=np.float32)
        ridx = self.regime_index_at(self.t)
        if 0 <= ridx < len(REGIME_LABELS):
            regime_onehot[ridx] = 1.0
        oracle_state = np.concatenate([base.state, regime_onehot])
        return ObservationView(state=oracle_state, feature_map=base.feature_map)

    def step(
        self, action: float | Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, float | int | str | list[float] | dict[str, float]]]:
        desired_position = self._normalize_action(action)
        current_prices = self.prices[self.t].astype(np.float64, copy=False)
        next_prices = self.prices[self.t + 1].astype(np.float64, copy=False)
        previous_position = self.positions.copy()
        trade_size = np.abs(desired_position - previous_position)

        transaction_cost = self.config.transaction_cost * trade_size
        hold_penalty = self.config.position_penalty * np.abs(desired_position)
        slippage_cost = (self.config.slippage_bps / 10_000.0) * trade_size

        self.entry_prices = self._updated_entry_prices(
            desired_position,
            previous_position,
            current_prices,
        )

        self.positions = desired_position
        simple_return = (next_prices / current_prices) - 1.0
        pnl_vector = desired_position * simple_return
        reward_vector = pnl_vector - transaction_cost - hold_penalty - slippage_cost
        reward = float(reward_vector.mean())

        current_regime = self.regimes[self.t]
        current_regime_index = self._regime_index.get(current_regime, -1)
        self.t += 1
        done = self.t >= self.end_index
        next_view = self.observe()

        return next_view.state, reward, done, {
            "step": self.t,
            "positions": desired_position.astype(np.float64).tolist(),
            "previous_positions": previous_position.astype(np.float64).tolist(),
            "prices": current_prices.astype(np.float64).tolist(),
            "next_prices": next_prices.astype(np.float64).tolist(),
            "reward": reward,
            "pnl": float(pnl_vector.mean()),
            "transaction_cost": float(transaction_cost.mean()),
            "hold_penalty": float(hold_penalty.mean()),
            "slippage_cost": float(slippage_cost.mean()),
            "reward_vector": reward_vector.astype(np.float64).tolist(),
            "regime": current_regime,
            "regime_index": current_regime_index,
            "feature_map": next_view.feature_map,
        }

    def _normalize_action(self, action: float | Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(action, dtype=np.float64).reshape(-1)
        if array.size == 1 and self.num_assets > 1:
            array = np.repeat(array.item(), self.num_assets)
        if array.size != self.num_assets:
            raise ValueError(
                f"Expected {self.num_assets} continuous action values, received shape {array.shape}."
            )
        return np.clip(array, self.action_space.low, self.action_space.high)

    def _updated_entry_prices(
        self,
        desired_position: np.ndarray,
        previous_position: np.ndarray,
        current_prices: np.ndarray,
    ) -> np.ndarray:
        updated = self.entry_prices.copy()
        tolerance = 1e-8

        for asset_idx in range(self.num_assets):
            target = float(desired_position[asset_idx])
            previous = float(previous_position[asset_idx])
            price = float(current_prices[asset_idx])
            previous_entry = float(updated[asset_idx])

            if np.isclose(target, 0.0, atol=tolerance):
                updated[asset_idx] = np.nan
                continue

            if np.isclose(previous, 0.0, atol=tolerance) or np.sign(target) != np.sign(previous):
                updated[asset_idx] = price
                continue

            if np.isnan(previous_entry):
                updated[asset_idx] = price
                continue

            if abs(target) <= abs(previous) + tolerance:
                updated[asset_idx] = previous_entry
                continue

            added_exposure = abs(target) - abs(previous)
            updated[asset_idx] = (
                (abs(previous) * previous_entry) + (added_exposure * price)
            ) / abs(target)

        return updated

    def _generate_regimes(self, steps: int) -> list[str]:
        regimes = [self.rng.choice(REGIME_LABELS, p=np.asarray([0.34, 0.18, 0.38, 0.10]))]
        for _ in range(1, steps):
            current_idx = self._regime_index[regimes[-1]]
            next_idx = int(self.rng.choice(len(REGIME_LABELS), p=self._transition_matrix[current_idx]))
            regimes.append(REGIME_LABELS[next_idx])
        return regimes

    def _generate_returns(self, regimes: list[str]) -> np.ndarray:
        returns = np.zeros((len(regimes), self.num_assets), dtype=np.float64)
        centered_asset_index = np.linspace(-0.5, 0.5, num=self.num_assets, dtype=np.float64)
        correlation = 0.65 if self.num_assets > 1 else 0.0
        common_scale = correlation
        idio_scale = float(np.sqrt(max(1.0 - (correlation**2), 1e-6)))

        for idx, regime in enumerate(regimes):
            params = self.config.regime_params[regime]
            drift_multiplier, vol_multiplier = self._nonstationary_adjustment(idx, len(regimes))
            base_vol = params["vol"] * vol_multiplier
            common_noise = self.rng.normal(0.0, base_vol)
            common_jump = 0.0
            if params["jump_prob"] > 0.0 and self.rng.random() < params["jump_prob"]:
                common_jump = self.rng.normal(0.0, params["jump_scale"] * vol_multiplier)

            for asset_idx in range(self.num_assets):
                previous = returns[idx - 1, asset_idx] if idx > 0 else 0.0
                idio_noise = self.rng.normal(0.0, base_vol * (0.9 + 0.2 * abs(centered_asset_index[asset_idx])))
                asset_drift = params["drift"] * drift_multiplier * (1.0 + 0.35 * centered_asset_index[asset_idx])
                asset_jump = common_jump * (1.0 + 0.15 * centered_asset_index[asset_idx])
                noise = (common_scale * common_noise) + (idio_scale * idio_noise)
                returns[idx, asset_idx] = (
                    asset_drift
                    + params["autocorr"] * previous
                    + noise
                    + asset_jump
                )
        return returns

    def _nonstationary_adjustment(self, idx: int, total_steps: int) -> tuple[float, float]:
        scale = max(float(self.config.nonstationary_drift_scale), 0.0)
        mode = str(self.config.nonstationary_mode).lower()
        if scale == 0.0 or mode == "stationary":
            return 1.0, 1.0

        phase = idx / max(total_steps - 1, 1)
        if mode in {"drift", "trend"}:
            return 1.0 + scale * np.sin(2.0 * np.pi * phase), 1.0
        if mode == "volatility":
            return 1.0, 1.0 + scale * (0.5 + 0.5 * np.sin(2.0 * np.pi * phase))
        if mode == "cyclical":
            return (
                1.0 + scale * np.sin(4.0 * np.pi * phase),
                1.0 + 0.5 * scale * (1.0 + np.cos(2.0 * np.pi * phase)),
            )
        if mode == "regime_shift":
            if phase < 0.33:
                return 1.0 - 0.5 * scale, 1.0
            if phase < 0.66:
                return 1.0 + scale, 1.0 + 0.25 * scale
            return 1.0 - scale, 1.0 + scale
        return 1.0 + 0.5 * scale * np.sin(2.0 * np.pi * phase), 1.0 + 0.25 * scale

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        return FEATURE_NAMES


class ContinuousMarketEnv(_BaseContinuousMarketEnv):
    def __init__(self, config: TrainingConfig, asset_name: str = "asset"):
        super().__init__(config=config, asset_names=(asset_name,))


class MultiAssetContinuousMarketEnv(_BaseContinuousMarketEnv):
    def __init__(self, config: TrainingConfig, asset_names: Sequence[str] | None = None):
        names = tuple(asset_names or config.real_data_symbols or ("asset_0", "asset_1"))
        super().__init__(config=config, asset_names=names)


def make_market_env(
    config: TrainingConfig,
    *,
    continuous: bool | None = None,
    multi_asset: bool = False,
    asset_names: Sequence[str] | None = None,
):
    use_continuous = config.continuous_actions if continuous is None else continuous
    if not use_continuous:
        return SyntheticMarketEnv(config)
    if multi_asset or (asset_names is not None and len(tuple(asset_names)) > 1):
        return MultiAssetContinuousMarketEnv(config, asset_names=asset_names)
    name = tuple(asset_names)[0] if asset_names else "asset"
    return ContinuousMarketEnv(config, asset_name=name)


__all__ = [
    "BoxSpace",
    "ContinuousMarketEnv",
    "MultiAssetContinuousMarketEnv",
    "make_market_env",
]
