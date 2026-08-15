from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingBuffer:
    channels: int
    samples: int

    def __post_init__(self) -> None:
        self._data = np.zeros((self.channels, self.samples), dtype=float)
        self._filled = 0

    def append(self, chunk: np.ndarray) -> None:
        n = chunk.shape[1]
        if n >= self.samples:
            self._data[:] = chunk[:, -self.samples :]
            self._filled = self.samples
            return
        self._data = np.roll(self._data, -n, axis=1)
        self._data[:, -n:] = chunk
        self._filled = min(self.samples, self._filled + n)

    def latest(self, n_samples: int) -> np.ndarray:
        if n_samples > self._filled:
            raise ValueError("not enough samples in ring buffer")
        return self._data[:, -n_samples:].copy()
