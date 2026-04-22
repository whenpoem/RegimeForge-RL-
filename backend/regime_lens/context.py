from __future__ import annotations

from collections import deque
from typing import Deque, Iterable

import numpy as np


def initialise_context_history(state: np.ndarray, *, context_len: int) -> Deque[np.ndarray]:
    history: Deque[np.ndarray] = deque(maxlen=context_len)
    for _ in range(max(1, context_len)):
        history.append(np.asarray(state, dtype=np.float32).copy())
    return history


def build_temporal_context(history: Iterable[np.ndarray], *, context_len: int) -> np.ndarray:
    items = [np.asarray(item, dtype=np.float32) for item in history]
    if not items:
        raise ValueError("Temporal context requires at least one observation.")
    while len(items) < context_len:
        items.insert(0, items[0].copy())
    if len(items) > context_len:
        items = items[-context_len:]
    return np.concatenate(items, axis=0).astype(np.float32, copy=False)


def append_context_state(history: Deque[np.ndarray], state: np.ndarray) -> Deque[np.ndarray]:
    updated = deque(history, maxlen=history.maxlen)
    updated.append(np.asarray(state, dtype=np.float32).copy())
    return updated
