from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import os

import torch


_PRIORITY_MAP = {
    "idle": 0x00000040,
    "below_normal": 0x00004000,
    "normal": 0x00000020,
}


@dataclass(slots=True)
class RuntimeInfo:
    requested_device: str
    resolved_device: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    cpu_threads: int
    process_priority: str
    priority_applied: bool

    def to_payload(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


def resolve_device(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device selection: {requested}")
    if normalized == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if normalized == "cuda":
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return "cpu"


def configure_runtime(requested_device: str, cpu_threads: int | None, process_priority: str) -> RuntimeInfo:
    resolved_device = resolve_device(requested_device)
    normalized_priority = process_priority.strip().lower()
    if normalized_priority not in _PRIORITY_MAP:
        normalized_priority = "normal"

    if resolved_device == "cpu":
        available = os.cpu_count() or 4
        target_threads = cpu_threads if cpu_threads is not None else max(1, min(4, available - 1))
        torch.set_num_threads(target_threads)
        try:
            torch.set_num_interop_threads(max(1, min(2, target_threads)))
        except RuntimeError:
            # Inter-op threads can only be set once per process.
            pass
    else:
        target_threads = torch.get_num_threads()
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    priority_applied = _apply_process_priority(normalized_priority)
    gpu_name = torch.cuda.get_device_name(0) if resolved_device == "cuda" and torch.cuda.is_available() else None
    return RuntimeInfo(
        requested_device=requested_device,
        resolved_device=resolved_device,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        gpu_name=gpu_name,
        cpu_threads=target_threads,
        process_priority=normalized_priority,
        priority_applied=priority_applied,
    )


def _apply_process_priority(priority_name: str) -> bool:
    if os.name != "nt":
        return False
    priority_class = _PRIORITY_MAP.get(priority_name)
    if priority_class is None:
        return False
    kernel32 = ctypes.windll.kernel32
    process_handle = kernel32.GetCurrentProcess()
    return bool(kernel32.SetPriorityClass(process_handle, priority_class))
