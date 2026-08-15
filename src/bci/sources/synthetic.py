from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from bci.domain import BCIEvent, EEGChunk, EEGMetadata
from bci.sources.base import EEGSource


class SyntheticEEGSource(EEGSource):
    def __init__(self, sfreq: float = 256.0, ch_names: list[str] | None = None):
        self.sfreq = sfreq
        self.ch_names = ch_names or ["Oz"]
        self._connected = False

    def connect(self) -> EEGMetadata:
        self._connected = True
        return EEGMetadata(self.sfreq, self.ch_names, "synthetic")

    def read_latest(self, seconds: float) -> EEGChunk:
        if not self._connected:
            raise RuntimeError("source is not connected")
        n = int(round(seconds * self.sfreq))
        t = np.arange(n) / self.sfreq
        data = np.sin(2 * np.pi * 13.0 * t)[None, :]
        return EEGChunk(data=data, sfreq=self.sfreq, ch_names=self.ch_names, t_start=0.0)

    def iter_events(self) -> Iterable[BCIEvent]:
        return []

    def close(self) -> None:
        self._connected = False
