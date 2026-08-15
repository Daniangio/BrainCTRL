from __future__ import annotations

from collections import Counter

from bci.domain import TrialRecord


class CalibrationPanel:
    def __init__(self):
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        self.label = QLabel("trials: 0 | phase: BOOTSTRAP | model v0")
        layout.addWidget(self.label)
        self.counts = Counter()
        self.phase = "BOOTSTRAP"
        self.model_version = 0

    def add_trial(self, trial: TrialRecord) -> None:
        self.counts[f"{trial.split}:{trial.command}"] += 1
        self._refresh()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self._refresh()

    def set_model(self, version: int, metrics: dict) -> None:
        self.model_version = version
        self._refresh()

    def _refresh(self) -> None:
        total = sum(self.counts.values())
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        self.label.setText(f"trials: {total} | phase: {self.phase} | model v{self.model_version} | {counts}")
