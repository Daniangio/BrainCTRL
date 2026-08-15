from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CalibrationPhase(str, Enum):
    READY = "READY"
    BOOTSTRAP = "BOOTSTRAP"
    CALIBRATING = "CALIBRATING"
    CALIBRATION_STREAMING = "CALIBRATION_STREAMING"
    CALIBRATION_FITTING = "CALIBRATION_FITTING"
    CALIBRATION_READY = "CALIBRATION_READY"
    VALIDATING = "VALIDATING"
    CHALLENGE_STREAMING = "CHALLENGE_STREAMING"
    CHALLENGE_REVIEW = "CHALLENGE_REVIEW"
    APPEND_CALIBRATION = "APPEND_CALIBRATION"
    REFITTING = "REFITTING"
    FINAL_TEST_READY = "FINAL_TEST_READY"
    FINAL_TEST_STREAMING = "FINAL_TEST_STREAMING"
    FROZEN_TEST = "FROZEN_TEST"
    INFERENCE = "INFERENCE"
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class RecordingRef:
    dataset: str
    subject: int
    session: str
    run: str


@dataclass(frozen=True)
class EEGMetadata:
    sfreq: float
    ch_names: list[str]
    source_name: str
    source_id: str | None = None


@dataclass(frozen=True)
class EEGChunk:
    data: np.ndarray
    sfreq: float
    ch_names: list[str]
    t_start: float
    times: np.ndarray | None = None
    annotations: list["BCIEvent"] = field(default_factory=list)


@dataclass(frozen=True)
class BCIEvent:
    timestamp: float
    duration: float
    native_label: str
    command: str | None
    event_index: int | None = None
    dataset: str | None = None
    subject: int | None = None
    session: str | None = None
    run: str | None = None


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    dataset: str
    subject: int
    session: str
    run: str
    event_index: int
    native_label: str
    command: str
    start_time: float
    end_time: float
    sfreq: float
    ch_names: list[str]
    data: np.ndarray
    source_event_id: int | None = None
    split: str | None = None
    feature_config_hash: str | None = None

    @property
    def provenance_key(self) -> tuple[str, int, str, str, int]:
        return (self.dataset, self.subject, self.session, self.run, self.event_index)


@dataclass(frozen=True)
class FeatureRecord:
    trial_id: str
    label: str
    split: str
    values: np.ndarray
    feature_names: list[str]
    frequency_scores: dict[str, float]
    provenance: dict[str, Any]
    config_hash: str
    spectral_freqs: np.ndarray | None = None
    log_power: np.ndarray | None = None
    spectral_power: np.ndarray | None = None
    spectral_channel_names: list[str] | None = None
    omitted_harmonics: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Prediction:
    trial_id: str
    true_label: str | None
    probabilities: dict[str, float]
    predicted_label: str
    confidence: float
    model_version: int
    timestamp: float


@dataclass(frozen=True)
class Decision:
    timestamp: float
    command: str
    probabilities: dict[str, float]
    confidence: float
    model_version: int
    reason: str = "unspecified"
    threshold: float | None = None
    consecutive: int = 0
    required_consecutive: int = 0


@dataclass
class CalibrationState:
    phase: CalibrationPhase = CalibrationPhase.BOOTSTRAP
    model_version: int = 0
    accumulated_trial_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecoderDiagnostics:
    model_version: int
    classes: list[str]
    latent_dim: int
    latent_points: np.ndarray | None
    latent_labels: list[str] | None
    class_centers: dict[str, np.ndarray]
    class_covariances: dict[str, np.ndarray]
    separation: dict[str, float]
