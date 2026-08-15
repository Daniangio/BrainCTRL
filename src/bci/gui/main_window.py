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
from bci.gui.panels.interpretation import InterpretationPanel
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
        self.interpretation = InterpretationPanel()
        self.signal = SignalPanel(config)
        self.spectrum = SpectrumPanel(config)
        self.probabilities = ProbabilityPanel()
        self.latent = LatentPanel()
        self.calibration = CalibrationPanel()
        self.controls = ControlsPanel(config, self._window.action_requested.emit)
        root = QWidget()
        layout = QGridLayout(root)
        layout.addWidget(self.status.widget, 0, 0, 1, 2)
        layout.addWidget(self.interpretation.widget, 1, 0, 1, 2)
        layout.addWidget(self.signal.widget, 2, 0)
        layout.addWidget(self.spectrum.widget, 2, 1)
        layout.addWidget(self.probabilities.widget, 3, 0)
        layout.addWidget(self.latent.widget, 3, 1)
        layout.addWidget(self.calibration.widget, 4, 0, 1, 2)
        layout.addWidget(self.controls.widget, 5, 0, 1, 2)
        self._window.setCentralWidget(root)

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
        elif isinstance(event, EEGWindowReady):
            self.signal.update_chunk(event.chunk)
        elif isinstance(event, TrialStarted):
            self.calibration.start_trial(event.event)
        elif isinstance(event, GroundTruthChanged):
            self.interpretation.update_ground_truth(event)
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
                self.interpretation.update_prediction(event.prediction)
            if event.decision is not None:
                self.probabilities.update_decision(event.decision)
                self.interpretation.update_decision(event.decision)
                self.status.set_decision(event.decision)
        elif isinstance(event, ModelUpdated):
            self.status.set_model(event.model_version)
            self.calibration.set_model(event.model_version, event.metrics)
            self.latent.update_diagnostics(event.diagnostics)
        elif isinstance(event, CalibrationStatus):
            self.calibration.set_status(event)
        elif isinstance(event, PredictionProduced):
            self.probabilities.update_prediction(event.prediction)
            self.interpretation.update_prediction(event.prediction)
        elif isinstance(event, DecisionEmitted):
            self.probabilities.update_decision(event.decision)
            self.interpretation.update_decision(event.decision)
            self.status.set_decision(event.decision)
        elif isinstance(event, ExperimentFinished):
            self.status.set_message(f"Finished: {event.artifact_dir}")

    def __getattr__(self, name):
        return getattr(self._window, name)
