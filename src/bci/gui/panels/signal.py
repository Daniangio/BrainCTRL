from __future__ import annotations

import numpy as np

from bci.config import BCIConfig
from bci.domain import EEGChunk


class SignalPanel:
    def __init__(self, config: BCIConfig):
        import pyqtgraph as pg
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        self.config = config
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        self.plot = pg.PlotWidget(title="EEG traces")
        self.plot.setLabel("bottom", "seconds")
        self.plot.setLabel("left", "amplitude + offset")
        layout.addWidget(self.plot)
        self.curves = []

    def update_chunk(self, chunk: EEGChunk) -> None:
        if not self.config.gui.show_raw_eeg:
            return
        data = chunk.data[: self.config.gui.max_channels_displayed]
        times = chunk.times if chunk.times is not None else chunk.t_start + np.arange(data.shape[1]) / chunk.sfreq
        if data.size == 0:
            return
        if len(self.curves) != data.shape[0]:
            self.plot.clear()
            self.curves = [self.plot.plot(pen=idx) for idx in range(data.shape[0])]
        scale = np.nanstd(data) or 1.0
        for idx, curve in enumerate(self.curves):
            curve.setData(times - times[-1], data[idx] / scale + idx * 3.0)
