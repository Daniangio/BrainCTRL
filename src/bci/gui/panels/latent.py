from __future__ import annotations


class LatentPanel:
    def __init__(self):
        import pyqtgraph as pg
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        self.plot = pg.PlotWidget(title="Latent diagnostics")
        layout.addWidget(self.plot)
        self.text = pg.TextItem("waiting for model")
        self.plot.addItem(self.text)

    def update_metrics(self, metrics: dict) -> None:
        separation = metrics.get("separation", {})
        message = "\n".join(f"{k}: {v:.2f}" for k, v in separation.items()) or "model updated"
        self.text.setText(message)
