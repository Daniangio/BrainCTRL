from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from bci.domain import BCIEvent, EEGChunk, EEGMetadata
from bci.sources.base import EEGSource, EventSource


class SyntheticEEGSource(EEGSource):
    def __init__(self, sfreq: float = 256.0, ch_names: list[str] | None = None):
        self.sfreq = sfreq
        self.ch_names = ch_names or ["Oz"]
        self._connected = False
        self._cursor = 0

    def connect(self) -> EEGMetadata:
        self._connected = True
        return EEGMetadata(self.sfreq, self.ch_names, "synthetic")

    def read_latest(self, seconds: float) -> EEGChunk:
        if not self._connected:
            raise RuntimeError("source is not connected")
        n = int(round(seconds * self.sfreq))
        t = (np.arange(n) + self._cursor) / self.sfreq
        data = np.sin(2 * np.pi * 13.0 * t)[None, :]
        self._cursor += n
        return EEGChunk(data=data, sfreq=self.sfreq, ch_names=self.ch_names, t_start=float(t[0]), times=t)

    def poll_new(self) -> EEGChunk | None:
        return self.read_latest(0.05)

    def iter_events(self) -> Iterable[BCIEvent]:
        return []

    def close(self) -> None:
        self._connected = False


class ScriptedSyntheticEEGSource(EEGSource):
    def __init__(self, data: np.ndarray, sfreq: float, ch_names: list[str], chunk_samples: int = 16):
        self.data = np.asarray(data, dtype=float)
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.chunk_samples = chunk_samples
        self.cursor = 0
        self._connected = False

    def connect(self) -> EEGMetadata:
        self.cursor = 0
        self._connected = True
        return EEGMetadata(self.sfreq, self.ch_names, "scripted-synthetic")

    def read_latest(self, seconds: float) -> EEGChunk:
        n = int(round(seconds * self.sfreq))
        return self._read(n)

    def poll_new(self) -> EEGChunk | None:
        if not self._connected:
            raise RuntimeError("source is not connected")
        if self.cursor >= self.data.shape[1]:
            return None
        return self._read(self.chunk_samples)

    def _read(self, n: int) -> EEGChunk | None:
        stop = min(self.cursor + n, self.data.shape[1])
        if stop <= self.cursor:
            return None
        start = self.cursor
        self.cursor = stop
        times = np.arange(start, stop) / self.sfreq
        return EEGChunk(
            data=self.data[:, start:stop],
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            t_start=float(times[0]),
            times=times,
        )

    def iter_events(self) -> Iterable[BCIEvent]:
        return []

    def close(self) -> None:
        self._connected = False


class SyntheticEventSource(EventSource):
    def __init__(self, events: list[BCIEvent]):
        self.events = sorted(events, key=lambda e: e.timestamp)
        self.index = 0
        self.current_time = 0.0

    def connect(self) -> None:
        self.index = 0
        self.current_time = 0.0

    def advance_to(self, timestamp: float) -> None:
        self.current_time = timestamp

    def poll(self) -> list[BCIEvent]:
        out: list[BCIEvent] = []
        while self.index < len(self.events) and self.events[self.index].timestamp <= self.current_time:
            out.append(self.events[self.index])
            self.index += 1
        return out

    def close(self) -> None:
        pass
