from __future__ import annotations

from collections import Counter

from bci.domain import BCIEvent, TrialRecord
from bci.experiment.events import CalibrationStatus


class CalibrationPanel:
    def __init__(self):
        from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        self.label = QLabel("trials: 0 | phase: BOOTSTRAP | model v0")
        self.current = QLabel("current: waiting for event")
        self.reason = QLabel("reason: collecting calibration examples")
        for label in [self.label, self.current, self.reason]:
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.label)
        layout.addWidget(self.current)
        layout.addWidget(self.reason)
        self.counts = Counter()
        self.phase = "BOOTSTRAP"
        self.model_version = 0
        self.batch_text = "batch: waiting"

    def start_trial(self, event: BCIEvent) -> None:
        self.current.setText(
            f"event {event.event_index}: stimulus {event.command} | native {event.native_label} | "
            f"window collecting"
        )

    def add_trial(self, trial: TrialRecord) -> None:
        self.counts[f"{trial.split}:{trial.command}"] += 1
        self._refresh()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self._refresh()

    def set_model(self, version: int, metrics: dict) -> None:
        self.model_version = version
        self._refresh()

    def set_batch(self, n_batch: int, n_total: int) -> None:
        self.batch_text = f"batch ready: +{n_batch}, total {n_total}"
        self._refresh()

    def set_status(self, status: CalibrationStatus) -> None:
        self.reason.setText(f"reason: {status.reason}")
        counts = ", ".join(f"{k} {v}/{status.required_per_class}" for k, v in status.counts.items())
        self.current.setText(f"calibration: {counts}")
        self.model_version = status.model_version
        self._refresh()

    def _refresh(self) -> None:
        total = sum(self.counts.values())
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        self.label.setText(f"trials: {total} | phase: {self.phase} | model v{self.model_version} | {self.batch_text} | {counts}")
