from __future__ import annotations

from bci.domain import Decision, Prediction


class ProbabilityPanel:
    def __init__(self):
        from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

        self.widget = QWidget()
        self.layout = QGridLayout(self.widget)
        self.bars: dict[str, QProgressBar] = {}
        self.labels: dict[str, QLabel] = {}
        self.decision = QLabel("decision: NONE")
        self.layout.addWidget(self.decision, 99, 0, 1, 2)

    def _ensure(self, classes):
        from PySide6.QtWidgets import QLabel, QProgressBar

        for row, cls in enumerate(classes):
            if cls in self.bars:
                continue
            label = QLabel(cls)
            bar = QProgressBar()
            bar.setRange(0, 100)
            self.layout.addWidget(label, row, 0)
            self.layout.addWidget(bar, row, 1)
            self.labels[cls] = label
            self.bars[cls] = bar

    def update_prediction(self, prediction: Prediction) -> None:
        self._ensure(prediction.probabilities)
        for cls, prob in prediction.probabilities.items():
            self.bars[cls].setValue(int(round(prob * 100)))

    def update_decision(self, decision: Decision) -> None:
        self.decision.setText(
            f"decision: {decision.command} | evidence {decision.confidence:.2f} | "
            f"{decision.reason} | {decision.consecutive}/{decision.required_consecutive}"
        )
