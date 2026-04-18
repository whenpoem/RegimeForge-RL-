from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ACTION_VALUES, FEATURE_NAMES, REGIME_LABELS, TrainingConfig


@dataclass(slots=True)
class ObservationView:
    state: np.ndarray
    feature_map: dict[str, float]


class SyntheticMarketEnv:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.total_steps = config.episode_length + config.warmup_steps + 1
        self._transition_matrix = np.asarray(config.regime_transition, dtype=np.float64)
        self._regime_index = {label: index for index, label in enumerate(REGIME_LABELS)}
        self.rng = np.random.default_rng(config.seed)
        self.prices = np.empty(0, dtype=np.float64)
        self.returns = np.empty(0, dtype=np.float64)
        self.regimes: list[str] = []
        self.t = config.warmup_steps
        self.end_index = self.t + config.episode_length
        self.position = 0
        self.entry_price: float | None = None
        self._regime_schedule: list[str] | None = None

    # ---- Regime schedule control (for curriculum learning) ----

    def set_regime_schedule(self, schedule: list[str] | None) -> None:
        """Override the Markov chain with a predetermined regime sequence.

        Pass *None* to revert to the natural Markov-chain sampling.
        """
        self._regime_schedule = schedule

    def regime_at(self, t: int) -> str:
        """Return the ground-truth regime label at time step *t*."""
        if 0 <= t < len(self.regimes):
            return self.regimes[t]
        return "unknown"

    def regime_index_at(self, t: int) -> int:
        """Return the integer regime index at time step *t*."""
        label = self.regime_at(t)
        return self._regime_index.get(label, -1)

    # ---- Core gym-like interface ----

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        if self._regime_schedule is not None and len(self._regime_schedule) >= self.total_steps:
            self.regimes = list(self._regime_schedule[: self.total_steps])
        else:
            self.regimes = self._generate_regimes(self.total_steps)
        self.returns = self._generate_returns(self.regimes)
        self.prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], self.returns])))
        self.t = self.config.warmup_steps
        self.end_index = self.t + self.config.episode_length
        self.position = 0
        self.entry_price = None
        return self.observe().state

    def observe(self) -> ObservationView:
        idx = self.t
        current_price = float(self.prices[idx])
        recent_returns = self.returns[:idx]
        short_return = float(np.mean(recent_returns[max(0, idx - 3) : idx]) * 100.0)
        medium_return = float(np.mean(recent_returns[max(0, idx - 12) : idx]) * 100.0)
        volatility = float(np.std(recent_returns[max(0, idx - 10) : idx]) * 100.0)
        short_ma = float(np.mean(self.prices[max(0, idx - 5) : idx + 1]))
        long_ma = float(np.mean(self.prices[max(0, idx - 20) : idx + 1]))
        trend_gap = ((short_ma - long_ma) / current_price) * 100.0
        running_peak = float(np.max(self.prices[: idx + 1]))
        drawdown = ((current_price / running_peak) - 1.0) * 100.0
        unrealized = 0.0
        if self.position != 0 and self.entry_price:
            unrealized = self.position * ((current_price / self.entry_price) - 1.0) * 100.0

        raw_features = {
            "short_return_pct": short_return,
            "medium_return_pct": medium_return,
            "volatility_pct": volatility,
            "trend_gap_pct": float(trend_gap),
            "drawdown_pct": float(drawdown),
            "position": float(self.position),
            "unrealized_pnl_pct": float(unrealized),
        }
        state = np.asarray(
            [
                np.clip(short_return / 0.45, -4.0, 4.0),
                np.clip(medium_return / 0.9, -4.0, 4.0),
                np.clip(volatility / 1.7, -4.0, 4.0),
                np.clip(trend_gap / 1.25, -4.0, 4.0),
                np.clip(drawdown / 4.5, -4.0, 4.0),
                float(self.position),
                np.clip(unrealized / 2.2, -4.0, 4.0),
            ],
            dtype=np.float32,
        )
        return ObservationView(state=state, feature_map=raw_features)

    def observe_oracle(self) -> ObservationView:
        """Return observation with regime one-hot appended (for Oracle DQN)."""
        base = self.observe()
        regime_onehot = np.zeros(len(REGIME_LABELS), dtype=np.float32)
        ridx = self.regime_index_at(self.t)
        if 0 <= ridx < len(REGIME_LABELS):
            regime_onehot[ridx] = 1.0
        oracle_state = np.concatenate([base.state, regime_onehot])
        return ObservationView(state=oracle_state, feature_map=base.feature_map)

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, float | int | str | dict[str, float]]]:
        current_price = float(self.prices[self.t])
        next_price = float(self.prices[self.t + 1])
        previous_position = self.position
        desired_position = ACTION_VALUES[action_index]
        trade_size = abs(desired_position - previous_position)
        transaction_cost = self.config.transaction_cost * trade_size
        hold_penalty = self.config.position_penalty * abs(desired_position)

        # Slippage cost (proportional to trade size)
        slippage_cost = (self.config.slippage_bps / 10_000.0) * trade_size

        if desired_position == 0:
            self.entry_price = None
        elif desired_position != previous_position:
            self.entry_price = current_price

        self.position = desired_position
        simple_return = (next_price / current_price) - 1.0
        pnl = desired_position * simple_return
        reward = pnl - transaction_cost - hold_penalty - slippage_cost

        current_regime = self.regimes[self.t]
        current_regime_index = self._regime_index.get(current_regime, -1)
        self.t += 1
        done = self.t >= self.end_index
        next_view = self.observe()
        return next_view.state, float(reward), done, {
            "step": self.t,
            "price": current_price,
            "next_price": next_price,
            "position": desired_position,
            "previous_position": previous_position,
            "action": action_index,
            "action_value": desired_position,
            "regime": current_regime,
            "regime_index": current_regime_index,
            "reward": float(reward),
            "pnl": float(pnl),
            "transaction_cost": float(transaction_cost),
            "hold_penalty": float(hold_penalty),
            "slippage_cost": float(slippage_cost),
            "feature_map": next_view.feature_map,
        }

    # ---- Regime generation ----

    def _generate_regimes(self, steps: int) -> list[str]:
        regimes = [self.rng.choice(REGIME_LABELS, p=np.asarray([0.34, 0.18, 0.38, 0.10]))]
        for _ in range(1, steps):
            current_idx = self._regime_index[regimes[-1]]
            next_idx = int(self.rng.choice(len(REGIME_LABELS), p=self._transition_matrix[current_idx]))
            regimes.append(REGIME_LABELS[next_idx])
        return regimes

    def _generate_returns(self, regimes: list[str]) -> np.ndarray:
        returns = np.zeros(len(regimes), dtype=np.float64)
        for idx, regime in enumerate(regimes):
            params = self.config.regime_params[regime]
            noise = self.rng.normal(0.0, params["vol"])
            jump = 0.0
            if params["jump_prob"] > 0.0 and self.rng.random() < params["jump_prob"]:
                jump = self.rng.normal(0.0, params["jump_scale"])
            previous = returns[idx - 1] if idx > 0 else 0.0
            returns[idx] = params["drift"] + params["autocorr"] * previous + noise + jump
        return returns

    def baseline_state(self, trend_gap_pct: float, volatility_pct: float) -> np.ndarray:
        short_return = trend_gap_pct * 0.72
        medium_return = trend_gap_pct
        drawdown = min(0.0, -abs(trend_gap_pct) * 0.6 - volatility_pct * 0.18)
        unrealized = trend_gap_pct * 0.25
        normalized = np.asarray(
            [
                np.clip(short_return / 0.45, -4.0, 4.0),
                np.clip(medium_return / 0.9, -4.0, 4.0),
                np.clip(volatility_pct / 1.7, -4.0, 4.0),
                np.clip(trend_gap_pct / 1.25, -4.0, 4.0),
                np.clip(drawdown / 4.5, -4.0, 4.0),
                0.0,
                np.clip(unrealized / 2.2, -4.0, 4.0),
            ],
            dtype=np.float32,
        )
        return normalized

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        return FEATURE_NAMES
