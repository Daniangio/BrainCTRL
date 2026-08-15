from __future__ import annotations

import numpy as np
from scipy import signal

from bci.config import BCIConfig
from bci.domain import EEGChunk, EEGMetadata


class StreamingPreprocessor:
    def __init__(self, config: BCIConfig):
        self.config = config
        self._sos: np.ndarray | None = None
        self._sos_zi: np.ndarray | None = None
        self._notch_ba: tuple[np.ndarray, np.ndarray] | None = None
        self._notch_zi: np.ndarray | None = None
        self._ch_names: list[str] = []
        self._sfreq: float | None = None

    def reset(self, metadata: EEGMetadata) -> None:
        self._ch_names = list(metadata.ch_names)
        self._sfreq = metadata.sfreq
        n_channels = len(metadata.ch_names)
        bp = self.config.preprocessing.bandpass_hz
        if bp is not None:
            low, high = bp
            nyq = metadata.sfreq / 2.0
            high = min(high, nyq * 0.98)
            if low > 0 and high > low:
                self._sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
                self._sos_zi = np.zeros((self._sos.shape[0], n_channels, 2), dtype=float)
            else:
                self._sos = None
                self._sos_zi = None
        notch = self.config.preprocessing.notch_hz
        if notch is not None and notch < metadata.sfreq / 2.0:
            self._notch_ba = signal.iirnotch(notch, Q=30.0, fs=metadata.sfreq)
            b, a = self._notch_ba
            self._notch_zi = np.zeros((n_channels, max(len(a), len(b)) - 1), dtype=float)
        else:
            self._notch_ba = None
            self._notch_zi = None

    def process_chunk(self, chunk: EEGChunk) -> EEGChunk:
        if self._sfreq is None:
            self.reset(EEGMetadata(chunk.sfreq, chunk.ch_names, "stream"))
        data = np.asarray(chunk.data, dtype=float).copy()
        if self._sos is not None and self._sos_zi is not None:
            data, self._sos_zi = signal.sosfilt(self._sos, data, axis=1, zi=self._sos_zi)
        if self._notch_ba is not None and self._notch_zi is not None:
            b, a = self._notch_ba
            data, self._notch_zi = signal.lfilter(b, a, data, axis=1, zi=self._notch_zi)
        return EEGChunk(
            data=data,
            sfreq=chunk.sfreq,
            ch_names=chunk.ch_names,
            t_start=chunk.t_start,
            times=chunk.times,
            annotations=chunk.annotations,
        )
