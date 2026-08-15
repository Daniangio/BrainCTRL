from __future__ import annotations


import numpy as np

from bci.domain import DecoderDiagnostics


class LatentPanel:
    def __init__(self):
        import pyqtgraph as pg
        from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

        self.widget = QWidget()
        self.widget.setMinimumWidth(0)
        self.widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self.widget)
        self.plot = pg.PlotWidget(title="Latent diagnostics")
        self.plot.setMinimumWidth(0)
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.plot)
        self.text = pg.TextItem("waiting for model")
        self.plot.addItem(self.text)
        self._live_curve = None
        self._live_label = None

    def update_metrics(self, metrics: dict) -> None:
        separation = metrics.get("separation", {})
        message = "\n".join(f"{k}: {v:.2f}" for k, v in separation.items()) or "model updated"
        self.text.setText(message)

    def update_diagnostics(self, diagnostics: DecoderDiagnostics | None) -> None:
        if diagnostics is None:
            return
        self.plot.clear()
        self._live_curve = None
        if diagnostics.latent_points is None or diagnostics.latent_labels is None:
            self.text.setText("model updated")
            self.plot.addItem(self.text)
            self._ensure_live_curve()
            return
        points = diagnostics.latent_points
        labels = np.asarray(diagnostics.latent_labels)
        colors = {"LEFT": "g", "RIGHT": "c", "NONE": "m"}
        if points.shape[1] == 1:
            xs = points[:, 0]
            ys = np.zeros_like(xs)
        else:
            xs = points[:, 0]
            ys = points[:, 1]
        for label in sorted(set(labels)):
            mask = labels == label
            self.plot.plot(xs[mask], ys[mask], pen=None, symbol="o", symbolBrush=colors.get(str(label), "w"), name=str(label))
        for label, center in diagnostics.class_centers.items():
            x = float(center[0])
            y = float(center[1]) if len(center) > 1 else 0.0
            self.plot.plot([x], [y], pen=None, symbol="x", symbolSize=14, symbolBrush=colors.get(label, "w"))
        self._ensure_live_curve()

    def update_live_point(self, point: list[float], predicted_label: str | None) -> None:
        if not point:
            return
        curve = self._ensure_live_curve()
        x = float(point[0])
        y = float(point[1]) if len(point) > 1 else 0.0
        colors = {"LEFT": "g", "RIGHT": "c", "NONE": "m", None: "y"}
        curve.setData([x], [y], symbolBrush=colors.get(predicted_label, "y"))

    def _ensure_live_curve(self):
        if self._live_curve is None:
            self._live_curve = self.plot.plot([], [], pen=None, symbol="star", symbolSize=18, symbolBrush="y")
        return self._live_curve
