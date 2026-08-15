from __future__ import annotations

import sys

from bci.config import BCIConfig
from bci.experiment.bus import EventBus
from bci.experiment.factory import build_realtime_experiment
from bci.gui.controller import ExperimentWorker
from bci.gui.main_window import MainWindow


def run_gui(config: BCIConfig) -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    config.experiment.manual_start = True
    if config.source.replay.speed <= 0:
        config.source.replay.speed = 1.0
    bus = EventBus()
    managed = build_realtime_experiment(config, bus=bus)
    window = MainWindow(config)
    worker = ExperimentWorker(managed, bus)
    worker.event_received.connect(window.handle_event)
    worker.finished.connect(window.handle_finished)
    window.stop_requested.connect(worker.stop)
    window.action_requested.connect(worker.request_action)
    window.show()
    worker.start()
    return int(app.exec())
