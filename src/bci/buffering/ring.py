from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bci.domain import EEGChunk


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


class TimestampedRingBuffer:
    def __init__(self, max_seconds: float, sfreq: float, ch_names: list[str]):
        self.max_samples = max(1, int(round(max_seconds * sfreq)))
        self.sfreq = sfreq
        self.ch_names = ch_names
        self._data = np.empty((len(ch_names), 0), dtype=float)
        self._times = np.empty((0,), dtype=float)

    @property
    def latest_time(self) -> float | None:
        return float(self._times[-1]) if self._times.size else None

    @property
    def earliest_time(self) -> float | None:
        return float(self._times[0]) if self._times.size else None

    def append(self, chunk: EEGChunk) -> None:
        if chunk.times is None:
            times = chunk.t_start + np.arange(chunk.data.shape[1]) / chunk.sfreq
        else:
            times = np.asarray(chunk.times, dtype=float)
        data = np.asarray(chunk.data, dtype=float)
        if data.shape[1] != times.size:
            raise ValueError("chunk data/time length mismatch")
        if data.shape[0] != len(self.ch_names):
            raise ValueError("chunk channel count does not match buffer metadata")
        if self._times.size and times.size and times[0] <= self._times[-1]:
            keep = times > self._times[-1]
            data = data[:, keep]
            times = times[keep]
        if times.size == 0:
            return
        self._data = np.concatenate([self._data, data], axis=1)
        self._times = np.concatenate([self._times, times])
        if self._times.size > self.max_samples:
            self._data = self._data[:, -self.max_samples :]
            self._times = self._times[-self.max_samples :]

    def has_interval(self, start: float, end: float) -> bool:
        if self._times.size == 0:
            return False
        return self._times[0] <= start and self._times[-1] >= end

    def slice(self, start: float, end: float, expected_samples: int | None = None) -> EEGChunk:
        if not self.has_interval(start, end):
            raise ValueError("requested interval is not fully buffered")
        mask = (self._times >= start) & (self._times < end)
        data = self._data[:, mask]
        times = self._times[mask]
        if expected_samples is not None and data.shape[1] != expected_samples:
            center = start + np.arange(expected_samples) / self.sfreq
            data = np.vstack([np.interp(center, self._times, ch) for ch in self._data])
            times = center
        return EEGChunk(data=data, sfreq=self.sfreq, ch_names=list(self.ch_names), t_start=float(times[0]), times=times)
