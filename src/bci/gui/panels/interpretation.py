from __future__ import annotations

from bci.domain import Decision, Prediction
from bci.experiment.events import GroundTruthChanged


def _symbol(command: str | None) -> str:
    if command == "LEFT":
        return "<-"
    if command == "RIGHT":
        return "->"
    if command == "NONE":
        return "."
    return "?"


class InterpretationPanel:
    def __init__(self):
        from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

        self.widget = QWidget()
        layout = QGridLayout(self.widget)
        self.truth = QLabel("Ground truth\n?\nunknown")
        self.model = QLabel("Model argmax\n?\nunavailable")
        self.controller = QLabel("Controller output\n.\nNONE")
        for idx, label in enumerate([self.truth, self.model, self.controller]):
            label.setMinimumHeight(84)
            label.setStyleSheet("font-size: 24px; font-weight: 600;")
            layout.addWidget(label, 0, idx)

    def update_ground_truth(self, event: GroundTruthChanged) -> None:
        self.truth.setText(f"Ground truth\n{_symbol(event.command)} {event.command or 'unknown'}\n{event.native_label}")

    def update_prediction(self, prediction: Prediction) -> None:
        self.model.setText(
            f"Model argmax\n{_symbol(prediction.predicted_label)} {prediction.predicted_label}\nP={prediction.confidence:.2f}"
        )

    def update_decision(self, decision: Decision) -> None:
        self.controller.setText(f"Controller output\n{_symbol(decision.command)} {decision.command}\n{decision.reason}")
