from __future__ import annotations

from bci.experiment.bus import EventBus
from bci.experiment.factory import ManagedExperiment
from bci.protocol.state_machine import ProtocolAction


class ExperimentWorker:
    def __init__(self, managed: ManagedExperiment, bus: EventBus):
        from PySide6.QtCore import QThread, Signal

        class _WorkerThread(QThread):
            event_received = Signal(object)

            def __init__(self, managed_experiment: ManagedExperiment, event_bus: EventBus):
                super().__init__()
                self.managed = managed_experiment
                event_bus.subscribe("*", self.event_received.emit)

            def run(self):
                self.managed.run()

            def stop(self):
                self.managed.stop()

        self._thread = _WorkerThread(managed, bus)

    @property
    def event_received(self):
        return self._thread.event_received

    @property
    def finished(self):
        return self._thread.finished

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._thread.stop()

    def request_action(self, action: ProtocolAction, payload: dict | None = None) -> None:
        self._thread.managed.engine.request_action(action, payload or {})
