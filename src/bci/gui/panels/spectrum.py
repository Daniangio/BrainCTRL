from __future__ import annotations

import numpy as np

from bci.config import BCIConfig
from bci.domain import FeatureRecord


class SpectrumPanel:
    def __init__(self, config: BCIConfig):
        import pyqtgraph as pg
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QVBoxLayout

        self.config = config
        self.widget = QFrame()
        self.widget.setObjectName("Panel")
        self.widget.setMinimumWidth(0)
        self.widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self.widget)
        header = QHBoxLayout()
        title = QLabel("Channel spectra")
        title.setObjectName("PanelTitle")
        self.summary = QLabel("waiting for spectral data")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.summary.setWordWrap(True)

        self.start_channel = QSpinBox()
        self.start_channel.setRange(1, 1)
        self.start_channel.setValue(1)
        self.start_channel.setToolTip("First channel to show.")
        self.visible_count = QSpinBox()
        self.visible_count.setRange(1, 16)
        self.visible_count.setValue(max(1, min(config.gui.max_channels_displayed, 8)))
        self.visible_count.setToolTip("Maximum number of channel spectra to draw.")
        self.start_channel.valueChanged.connect(self._redraw_cached)
        self.visible_count.valueChanged.connect(self._redraw_cached)

        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QLabel("start"))
        header.addWidget(self.start_channel)
        header.addWidget(QLabel("show"))
        header.addWidget(self.visible_count)
        layout.addLayout(header)

        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setBackground("#111820")
        self.plots.setMinimumWidth(0)
        self.plots.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.plots, 1)
        layout.addWidget(self.summary)

        self._curves = []
        self._plot_channels: list[str] = []
        self._freqs: np.ndarray | None = None
        self._power: np.ndarray | None = None
        self._channels: list[str] = []

    def update_feature(self, feature: FeatureRecord) -> None:
        if feature.spectral_freqs is None:
            self.summary.setText("no spectral representation available")
            return
        power = self._feature_power(feature)
        if power is None or power.ndim != 2 or power.shape[1] != feature.spectral_freqs.size:
            self.summary.setText("spectral representation has incompatible shape")
            return
        self._freqs = np.asarray(feature.spectral_freqs, dtype=float)
        self._power = np.asarray(power, dtype=float)
        self._channels = feature.spectral_channel_names or [f"ch {idx + 1}" for idx in range(self._power.shape[0])]
        self.start_channel.setMaximum(max(1, len(self._channels)))
        self._redraw_cached()

    def _feature_power(self, feature: FeatureRecord) -> np.ndarray | None:
        if feature.spectral_power is not None:
            return np.asarray(feature.spectral_power, dtype=float)
        if feature.log_power is None:
            return None
        log_power = np.asarray(feature.log_power, dtype=float)
        if log_power.ndim == 1:
            log_power = log_power[None, :]
        return np.exp(log_power)

    def _redraw_cached(self) -> None:
        if self._freqs is None or self._power is None:
            return
        max_hz = self.config.gui.spectrum_max_hz
        mask = self._freqs <= max_hz
        freqs = self._freqs[mask]
        power = self._power[:, mask]
        channel_indices = self._visible_channel_indices()
        channel_names = [self._channels[idx] for idx in channel_indices]
        self._ensure_plots(channel_names)
        eps = self.config.features.log_epsilon
        db = 10.0 * np.log10(power[channel_indices] + eps)
        for curve, values in zip(self._curves, db):
            curve.setData(freqs, values)
        if channel_names:
            self.summary.setText(f"{len(channel_names)}/{len(self._channels)} channels | 0-{max_hz:g} Hz | PSD dB")

    def _visible_channel_indices(self) -> list[int]:
        start = max(0, self.start_channel.value() - 1)
        stop = min(len(self._channels), start + self.visible_count.value())
        return list(range(start, stop))

    def _ensure_plots(self, channel_names: list[str]) -> None:
        if channel_names == self._plot_channels:
            return
        self.plots.clear()
        self._curves = []
        self._plot_channels = list(channel_names)
        linked = None
        for row, channel in enumerate(channel_names):
            plot = self.plots.addPlot(row=row, col=0, title=channel)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", "dB")
            if row == len(channel_names) - 1:
                plot.setLabel("bottom", "Hz")
            else:
                plot.hideAxis("bottom")
            if linked is not None:
                plot.setXLink(linked)
            else:
                linked = plot
            curve = plot.plot(pen=(80, 170, 220), antialias=True)
            self._curves.append(curve)
