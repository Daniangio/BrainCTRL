from __future__ import annotations

from bci.config import BCIConfig
from bci.experiment.events import (
    DecisionEmitted,
    EEGWindowReady,
    ExperimentFinished,
    FeatureComputed,
    ModelUpdated,
    PhaseChanged,
    PredictionProduced,
    StreamConnected,
    TrialCompleted,
)
from bci.gui.panels.calibration import CalibrationPanel
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

            def closeEvent(self, event):  # noqa: N802
                self.stop_requested.emit()
                super().closeEvent(event)

        self._window = _Window()
        self._window.setWindowTitle("BrainCTRL realtime experiment")
        self.status = StatusPanel(config)
        self.signal = SignalPanel(config)
        self.spectrum = SpectrumPanel(config)
        self.probabilities = ProbabilityPanel()
        self.latent = LatentPanel()
        self.calibration = CalibrationPanel()
        root = QWidget()
        layout = QGridLayout(root)
        layout.addWidget(self.status.widget, 0, 0, 1, 2)
        layout.addWidget(self.signal.widget, 1, 0)
        layout.addWidget(self.spectrum.widget, 1, 1)
        layout.addWidget(self.probabilities.widget, 2, 0)
        layout.addWidget(self.latent.widget, 2, 1)
        layout.addWidget(self.calibration.widget, 3, 0, 1, 2)
        self._window.setCentralWidget(root)

    @property
    def stop_requested(self):
        return self._window.stop_requested

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
        elif isinstance(event, EEGWindowReady):
            self.signal.update_chunk(event.chunk)
        elif isinstance(event, TrialCompleted):
            self.calibration.add_trial(event.trial)
        elif isinstance(event, FeatureComputed):
            self.spectrum.update_feature(event.feature)
        elif isinstance(event, ModelUpdated):
            self.status.set_model(event.model_version)
            self.calibration.set_model(event.model_version, event.metrics)
            self.latent.update_metrics(event.metrics)
        elif isinstance(event, PredictionProduced):
            self.probabilities.update_prediction(event.prediction)
        elif isinstance(event, DecisionEmitted):
            self.probabilities.update_decision(event.decision)
            self.status.set_decision(event.decision)
        elif isinstance(event, ExperimentFinished):
            self.status.set_message(f"Finished: {event.artifact_dir}")

    def __getattr__(self, name):
        return getattr(self._window, name)
