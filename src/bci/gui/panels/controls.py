from __future__ import annotations

from collections.abc import Callable

from bci.config import BCIConfig
from bci.protocol.state_machine import ProtocolAction


class ControlsPanel:
    def __init__(self, config: BCIConfig, emit_action: Callable[[ProtocolAction, dict], None]):
        from PySide6.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QPushButton, QSpinBox, QWidget

        self._emit_action = emit_action
        self.widget = QWidget()
        layout = QGridLayout(self.widget)
        self.start_calibration = QPushButton("Start Calibration")
        self.start_challenge = QPushButton("Start Challenge")
        self.final_test = QPushButton("Final Test")
        self.pause = QPushButton("Pause")
        self.resume = QPushButton("Resume")
        self.step = QPushButton("Step")
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.0, 5.0)
        self.speed.setSingleStep(0.25)
        self.speed.setValue(config.source.replay.speed)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(config.decision.posterior_threshold)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 1.0)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(config.decision.alpha)
        self.consecutive = QSpinBox()
        self.consecutive.setRange(1, 10)
        self.consecutive.setValue(config.decision.consecutive_windows)
        self.refractory = QDoubleSpinBox()
        self.refractory.setRange(0.0, 10.0)
        self.refractory.setSingleStep(0.1)
        self.refractory.setValue(config.decision.refractory_seconds)

        for col, button in enumerate(
            [self.start_calibration, self.start_challenge, self.final_test, self.pause, self.resume, self.step]
        ):
            layout.addWidget(button, 0, col)
        for col, (label, widget) in enumerate(
            [
                ("speed", self.speed),
                ("threshold", self.threshold),
                ("alpha", self.alpha),
                ("consecutive", self.consecutive),
                ("refractory", self.refractory),
            ]
        ):
            layout.addWidget(QLabel(label), 1, col * 2)
            layout.addWidget(widget, 1, col * 2 + 1)

        self.start_calibration.clicked.connect(lambda: emit_action(ProtocolAction.START_CALIBRATION, {}))
        self.start_challenge.clicked.connect(lambda: emit_action(ProtocolAction.START_CHALLENGE, {}))
        self.final_test.clicked.connect(lambda: emit_action(ProtocolAction.START_FINAL_TEST, {}))
        self.pause.clicked.connect(lambda: emit_action(ProtocolAction.PAUSE, {}))
        self.resume.clicked.connect(lambda: emit_action(ProtocolAction.RESUME, {}))
        self.step.clicked.connect(lambda: emit_action(ProtocolAction.STEP, {}))
        self.speed.valueChanged.connect(lambda value: emit_action(ProtocolAction.SET_SPEED, {"speed": float(value)}))
        for widget in [self.threshold, self.alpha, self.consecutive, self.refractory]:
            widget.valueChanged.connect(self._emit_decision_params)

    def _emit_decision_params(self) -> None:
        self._emit_action(
            ProtocolAction.UPDATE_DECISION_PARAMS,
            {
                "posterior_threshold": float(self.threshold.value()),
                "alpha": float(self.alpha.value()),
                "consecutive_windows": int(self.consecutive.value()),
                "refractory_seconds": float(self.refractory.value()),
            },
        )
