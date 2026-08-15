from __future__ import annotations

import time

import numpy as np

from bci.config import BCIConfig
from bci.domain import EEGChunk


class SignalPanel:
    def __init__(self, config: BCIConfig):
        import pyqtgraph as pg
        from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

        self.config = config
        self.widget = QWidget()
        self.widget.setMinimumWidth(0)
        self.widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self.widget)
        self.plot = pg.PlotWidget(title="Raw EEG traces")
        self.plot.setBackground("#111820")
        self.plot.setMinimumWidth(0)
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot.setLabel("bottom", "seconds")
        self.plot.setLabel("left", "amplitude + offset")
        layout.addWidget(self.plot)
        self.curves = []
        self._data = None
        self._times = None
        self._last_draw_monotonic = 0.0

    def update_chunk(self, chunk: EEGChunk) -> None:
        if not self.config.gui.show_raw_eeg:
            return
        incoming = chunk.data[: self.config.gui.max_channels_displayed]
        incoming_times = chunk.times if chunk.times is not None else chunk.t_start + np.arange(incoming.shape[1]) / chunk.sfreq
        if self._data is None:
            self._data = incoming.copy()
            self._times = np.asarray(incoming_times, dtype=float).copy()
        else:
            self._data = np.concatenate([self._data, incoming], axis=1)
            self._times = np.concatenate([self._times, np.asarray(incoming_times, dtype=float)])
        order = np.argsort(self._times)
        self._times = self._times[order]
        self._data = self._data[:, order]
        unique = np.concatenate(([True], np.diff(self._times) > 1e-9))
        self._times = self._times[unique]
        self._data = self._data[:, unique]
        latest = float(self._times[-1])
        keep = self._times >= latest - self.config.gui.eeg_history_seconds
        self._data = self._data[:, keep]
        self._times = self._times[keep]
        data = self._data
        times = self._times
        if data.size == 0:
            return
        now = time.monotonic()
        min_interval = 1.0 / max(self.config.gui.refresh_hz, 1.0)
        if now - self._last_draw_monotonic < min_interval:
            return
        self._last_draw_monotonic = now
        if len(self.curves) != data.shape[0]:
            self.plot.clear()
            self.curves = [self.plot.plot(pen=idx) for idx in range(data.shape[0])]
        scale = np.nanstd(data) or 1.0
        for idx, curve in enumerate(self.curves):
            curve.setData(times - times[-1], data[idx] / scale + idx * 3.0)
