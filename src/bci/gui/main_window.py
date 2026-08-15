from __future__ import annotations

from bci.config import BCIConfig
from bci.experiment.events import (
    DecisionEmitted,
    EEGWindowReady,
    ExperimentFinished,
    FeatureComputed,
    CalibrationBatchReady,
    CalibrationStatus,
    GroundTruthChanged,
    InferenceUpdated,
    LiveWindowUpdated,
    ModelUpdated,
    PhaseChanged,
    PredictionProduced,
    StreamConnected,
    TrialCompleted,
    TrialStarted,
)
from bci.gui.panels.calibration import CalibrationPanel
from bci.gui.panels.controls import ControlsPanel
from bci.gui.panels.ground_truth import GroundTruthPanel
from bci.gui.panels.latent import LatentPanel
from bci.gui.panels.probabilities import ProbabilityPanel
from bci.gui.panels.signal import SignalPanel
from bci.gui.panels.spectrum import SpectrumPanel
from bci.gui.panels.status import StatusPanel


class MainWindow:
    def __init__(self, config: BCIConfig):
        from PySide6.QtCore import Signal
        from PySide6.QtWidgets import QGridLayout, QMainWindow, QWidget

        class _Window(QMainWindow):
            stop_requested = Signal()
            action_requested = Signal(object, object)

            def closeEvent(self, event):  # noqa: N802
                self.stop_requested.emit()
                super().closeEvent(event)

        self._window = _Window()
        self._window.setWindowTitle("BrainCTRL realtime experiment")
        self.status = StatusPanel(config)
        self.ground_truth = GroundTruthPanel()
        self.signal = SignalPanel(config)
        self.spectrum = SpectrumPanel(config)
        self.probabilities = ProbabilityPanel()
        self.latent = LatentPanel()
        self.calibration = CalibrationPanel()
        self.controls = ControlsPanel(config, self._window.action_requested.emit)
        for panel in [
            self.status.widget,
            self.ground_truth.widget,
            self.signal.widget,
            self.probabilities.widget,
            self.latent.widget,
            self.calibration.widget,
            self.controls.widget,
        ]:
            panel.setObjectName("Panel")
        root = QWidget()
        root.setObjectName("Root")
        layout = QGridLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 3)
        layout.setRowStretch(1, 4)
        layout.setRowStretch(2, 3)
        layout.setRowStretch(3, 3)
        layout.addWidget(self.status.widget, 0, 0, 1, 2)
        layout.addWidget(self.signal.widget, 1, 0)
        layout.addWidget(self.ground_truth.widget, 1, 1)
        layout.addWidget(self.latent.widget, 2, 0, 1, 2)
        layout.addWidget(self.probabilities.widget, 3, 0, 1, 2)
        layout.addWidget(self.calibration.widget, 4, 0, 1, 2)
        layout.addWidget(self.controls.widget, 5, 0, 1, 2)
        layout.addWidget(self.spectrum.widget, 0, 2, 6, 1)
        self._window.setCentralWidget(root)
        self._window.setStyleSheet(
            """
            QWidget#Root {
                background: #0f1419;
                color: #d7dde5;
                font-size: 12px;
            }
            QFrame#Panel, QWidget#Panel {
                background: #171d24;
                border: 1px solid #2a333d;
                border-radius: 8px;
            }
            QLabel {
                color: #d7dde5;
            }
            QLabel#PanelTitle {
                color: #f0f4f8;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton {
                background: #243241;
                border: 1px solid #3a4a5c;
                border-radius: 6px;
                color: #eef3f7;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background: #2f4052;
            }
            QPushButton:disabled {
                background: #1a2027;
                color: #697582;
                border-color: #28313a;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #10161d;
                border: 1px solid #354352;
                border-radius: 5px;
                color: #eef3f7;
                padding: 4px 6px;
            }
            QProgressBar {
                background: #10161d;
                border: 1px solid #354352;
                border-radius: 5px;
                color: #eef3f7;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #4aa3df;
                border-radius: 4px;
            }
            """
        )

    @property
    def stop_requested(self):
        return self._window.stop_requested

    @property
    def action_requested(self):
        return self._window.action_requested

    def show(self) -> None:
        self._window.resize(1280, 850)
        self._window.show()

    def handle_finished(self) -> None:
        self.status.set_message("Experiment worker finished")

    def handle_event(self, event: object) -> None:
        if isinstance(event, StreamConnected):
            self.status.set_stream(event.metadata)
        elif isinstance(event, PhaseChanged):
            self.status.set_phase(event.new_phase.value)
            self.calibration.set_phase(event.new_phase.value)
            self.controls.set_phase(event.new_phase.value)
        elif isinstance(event, EEGWindowReady):
            self.signal.update_chunk(event.chunk)
        elif isinstance(event, TrialStarted):
            self.calibration.start_trial(event.event)
        elif isinstance(event, GroundTruthChanged):
            self.ground_truth.update_ground_truth(event)
        elif isinstance(event, CalibrationBatchReady):
            self.calibration.set_batch(event.n_batch, event.n_total)
        elif isinstance(event, TrialCompleted):
            self.calibration.add_trial(event.trial)
        elif isinstance(event, FeatureComputed):
            self.spectrum.update_feature(event.feature)
        elif isinstance(event, LiveWindowUpdated):
            self.spectrum.update_feature(event.feature)
            if event.latent_point is not None:
                predicted_label = event.prediction.predicted_label if event.prediction is not None else None
                self.latent.update_live_point(event.latent_point, predicted_label)
            if event.prediction is not None:
                self.probabilities.update_prediction(event.prediction)
            if event.decision is not None:
                self.probabilities.update_decision(event.decision)
                self.status.set_decision(event.decision)
        elif isinstance(event, ModelUpdated):
            self.status.set_model(event.model_version)
            self.calibration.set_model(event.model_version, event.metrics)
            self.controls.set_model(event.model_version)
            self.latent.update_diagnostics(event.diagnostics)
        elif isinstance(event, CalibrationStatus):
            self.calibration.set_status(event)
        elif isinstance(event, PredictionProduced):
            self.probabilities.update_prediction(event.prediction)
        elif isinstance(event, DecisionEmitted):
            self.probabilities.update_decision(event.decision)
            self.status.set_decision(event.decision)
        elif isinstance(event, InferenceUpdated):
            if event.latent_point is not None:
                self.latent.update_live_point(event.latent_point, event.prediction.predicted_label)
            self.spectrum.update_feature(event.feature)
        elif isinstance(event, ExperimentFinished):
            self.status.set_message(f"Finished: {event.artifact_dir}")

    def __getattr__(self, name):
        return getattr(self._window, name)
