from __future__ import annotations

from bci.domain import CalibrationPhase, CalibrationState


class CalibrationStateMachine:
    def __init__(self):
        self.state = CalibrationState()

    def set_phase(self, phase: CalibrationPhase) -> None:
        self.state.phase = phase
