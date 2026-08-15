from __future__ import annotations

from bci.config import BCIConfig
from bci.domain import Decision, EEGMetadata


class StatusPanel:
    def __init__(self, config: BCIConfig):
        from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget, QHBoxLayout

        self.widget = QWidget()
        layout = QHBoxLayout(self.widget)
        self.label = QLabel(f"BrainCTRL | {config.dataset.name} | subject {config.dataset.subjects}")
        self.phase = QLabel("phase: BOOTSTRAP")
        self.model = QLabel("model v0")
        self.decision = QLabel("last command: NONE")
        for label in [self.label, self.phase, self.model, self.decision]:
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.label)
        layout.addWidget(self.phase)
        layout.addWidget(self.model)
        layout.addWidget(self.decision)

    def set_stream(self, metadata: EEGMetadata) -> None:
        self.label.setText(f"BrainCTRL | {metadata.source_name} | {len(metadata.ch_names)} ch @ {metadata.sfreq:g} Hz")

    def set_phase(self, phase: str) -> None:
        self.phase.setText(f"phase: {phase}")

    def set_model(self, version: int) -> None:
        self.model.setText(f"model v{version}")

    def set_decision(self, decision: Decision) -> None:
        self.decision.setText(f"last command: {decision.command} ({decision.confidence:.2f})")

    def set_message(self, message: str) -> None:
        self.decision.setText(message)
