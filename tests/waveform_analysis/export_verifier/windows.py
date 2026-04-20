"""Window extraction helpers for the experimental export verifier."""

from __future__ import annotations

import numpy as np


def tail(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = max(1, int(round(seconds * sr)))
    if len(y) >= n:
        return y[-n:]
    return np.pad(y, (n - len(y), 0))
