from __future__ import annotations

from bci.experiment.events import GroundTruthChanged


def _symbol(command: str | None) -> str:
    if command == "LEFT":
        return "<-"
    if command == "RIGHT":
        return "->"
    if command == "NONE":
        return "."
    return "?"


class GroundTruthPanel:
    def __init__(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

        self.widget = QWidget()
        self.widget.setFixedWidth(132)
        self.widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(10, 10, 10, 10)
        title = QLabel("Ground truth")
        title.setObjectName("PanelTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.command = QLabel("?")
        self.command.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.command.setStyleSheet("font-size: 30px; font-weight: 800;")
        self.native = QLabel("UNKNOWN")
        self.native.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.native.setWordWrap(True)
        self.native.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.command)
        layout.addWidget(self.native)
        layout.addStretch(1)

    def update_ground_truth(self, event: GroundTruthChanged) -> None:
        command = event.command or "UNKNOWN"
        self.command.setText(f"{_symbol(event.command)}\n{command}")
        self.native.setText(str(event.native_label))
