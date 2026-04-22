from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .config import TrackingBackend, TrainingConfig


class Tracker(Protocol):
    def log_episode(self, episode: int, payload: dict[str, Any]) -> None: ...

    def log_checkpoint(self, episode: int, payload: dict[str, Any]) -> None: ...

    def close(self) -> None: ...

    def status_payload(self) -> dict[str, Any]: ...


class NullTracker:
    def __init__(
        self,
        *,
        requested_backend: str = TrackingBackend.NONE.value,
        actual_backend: str = TrackingBackend.NONE.value,
        warning: str | None = None,
    ):
        self.requested_backend = requested_backend
        self.actual_backend = actual_backend
        self.warning = warning

    def log_episode(self, episode: int, payload: dict[str, Any]) -> None:
        return None

    def log_checkpoint(self, episode: int, payload: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None

    def status_payload(self) -> dict[str, Any]:
        payload = {
            "requestedBackend": self.requested_backend,
            "actualBackend": self.actual_backend,
            "fallback": self.requested_backend != self.actual_backend,
        }
        if self.warning:
            payload["warning"] = self.warning
        return payload


class TensorBoardTracker:
    def __init__(self, directory: Path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(directory))
        self.requested_backend = TrackingBackend.TENSORBOARD.value
        self.actual_backend = TrackingBackend.TENSORBOARD.value

    def log_episode(self, episode: int, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            if isinstance(value, (int, float)) and value is not None:
                self.writer.add_scalar(f"episode/{key}", float(value), episode)

    def log_checkpoint(self, episode: int, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            if isinstance(value, (int, float)) and value is not None:
                self.writer.add_scalar(f"checkpoint/{key}", float(value), episode)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def status_payload(self) -> dict[str, Any]:
        return {
            "requestedBackend": self.requested_backend,
            "actualBackend": self.actual_backend,
            "fallback": False,
        }


class WandBTracker:
    def __init__(self, config: TrainingConfig, directory: Path):
        import wandb

        self._wandb = wandb
        self.requested_backend = TrackingBackend.WANDB.value
        self.actual_backend = TrackingBackend.WANDB.value
        self._run = wandb.init(
            project="regimeforge",
            dir=str(directory),
            name=config.experiment_name,
            config={"agent_type": config.agent_type.value, "algorithm": config.algorithm.value},
            reinit=True,
        )

    def log_episode(self, episode: int, payload: dict[str, Any]) -> None:
        self._wandb.log({f"episode/{key}": value for key, value in payload.items()}, step=episode)

    def log_checkpoint(self, episode: int, payload: dict[str, Any]) -> None:
        self._wandb.log({f"checkpoint/{key}": value for key, value in payload.items()}, step=episode)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()

    def status_payload(self) -> dict[str, Any]:
        return {
            "requestedBackend": self.requested_backend,
            "actualBackend": self.actual_backend,
            "fallback": False,
        }


def create_tracker(config: TrainingConfig, run_dir: Path) -> Tracker:
    tracking_dir = run_dir / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    backend = config.tracking_backend
    if backend == TrackingBackend.NONE:
        return NullTracker()
    if backend == TrackingBackend.WANDB:
        try:
            return WandBTracker(config, tracking_dir)
        except Exception as exc:
            return NullTracker(
                requested_backend=TrackingBackend.WANDB.value,
                warning=f"W&B tracker unavailable: {type(exc).__name__}: {exc}",
            )
    try:
        return TensorBoardTracker(tracking_dir)
    except Exception as exc:
        return NullTracker(
            requested_backend=TrackingBackend.TENSORBOARD.value,
            warning=f"TensorBoard tracker unavailable: {type(exc).__name__}: {exc}",
        )
