from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from bci.domain import CalibrationPhase


class ProtocolAction(str, Enum):
    START_CALIBRATION = "START_CALIBRATION"
    START_CHALLENGE = "START_CHALLENGE"
    ADD_CALIBRATION = "ADD_CALIBRATION"
    START_FINAL_TEST = "START_FINAL_TEST"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    STEP = "STEP"
    SET_SPEED = "SET_SPEED"
    UPDATE_DECISION_PARAMS = "UPDATE_DECISION_PARAMS"


@dataclass(frozen=True)
class ProtocolCommand:
    action: ProtocolAction
    payload: dict | None = None


class ProtocolStateMachine:
    def __init__(self, initial: CalibrationPhase = CalibrationPhase.READY):
        self.state = initial

    def transition(self, action: ProtocolAction) -> CalibrationPhase:
        if action == ProtocolAction.START_CALIBRATION and self.state in {
            CalibrationPhase.READY,
            CalibrationPhase.BOOTSTRAP,
            CalibrationPhase.CALIBRATION_READY,
        }:
            self.state = CalibrationPhase.CALIBRATION_STREAMING
        elif action == ProtocolAction.ADD_CALIBRATION and self.state in {
            CalibrationPhase.CHALLENGE_REVIEW,
            CalibrationPhase.CALIBRATION_READY,
        }:
            self.state = CalibrationPhase.APPEND_CALIBRATION
        elif action == ProtocolAction.START_CHALLENGE and self.state in {
            CalibrationPhase.CALIBRATION_READY,
            CalibrationPhase.CHALLENGE_REVIEW,
        }:
            self.state = CalibrationPhase.CHALLENGE_STREAMING
        elif action == ProtocolAction.START_FINAL_TEST and self.state in {
            CalibrationPhase.CALIBRATION_READY,
            CalibrationPhase.CHALLENGE_REVIEW,
            CalibrationPhase.FINAL_TEST_READY,
        }:
            self.state = CalibrationPhase.FINAL_TEST_STREAMING
        return self.state
