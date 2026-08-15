from __future__ import annotations

from bci.config import BCIConfig
from bci.domain import FeatureRecord


class SpectrumPanel:
    def __init__(self, config: BCIConfig):
        import pyqtgraph as pg
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        self.config = config
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        self.plot = pg.PlotWidget(title="SSVEP spectral evidence")
        self.plot.setLabel("bottom", "feature index")
        self.plot.setLabel("left", "local log-SNR")
        layout.addWidget(self.plot)
        self.curve = self.plot.plot(pen="y")

    def update_feature(self, feature: FeatureRecord) -> None:
        self.curve.setData(feature.values)
