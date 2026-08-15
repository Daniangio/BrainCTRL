from __future__ import annotations

from bci.config import BCIConfig
from bci.domain import FeatureRecord


class SpectrumPanel:
    def __init__(self, config: BCIConfig):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        self.config = config
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        self.plot = pg.PlotWidget(title="SSVEP spectral evidence")
        self.plot.setLabel("bottom", "frequency [Hz]")
        self.plot.setLabel("left", "mean log power")
        layout.addWidget(self.plot)
        self.curve = self.plot.plot(pen="y")
        for freq in sorted(config.stimulus_frequencies.values()):
            for harmonic in config.features.harmonics:
                target = freq * harmonic
                if target <= config.gui.spectrum_max_hz:
                    line = pg.InfiniteLine(target, angle=90, pen=pg.mkPen("r", style=QtCore.Qt.PenStyle.DotLine))
                    self.plot.addItem(line)

    def update_feature(self, feature: FeatureRecord) -> None:
        if feature.spectral_freqs is not None and feature.log_power is not None:
            mask = feature.spectral_freqs <= self.config.gui.spectrum_max_hz
            self.curve.setData(feature.spectral_freqs[mask], feature.log_power[mask])
        else:
            self.curve.setData(feature.values)
