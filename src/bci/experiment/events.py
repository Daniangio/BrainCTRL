from __future__ import annotations

from dataclasses import dataclass

from bci.domain import BCIEvent, CalibrationPhase, Decision, EEGChunk, EEGMetadata, FeatureRecord, Prediction, TrialRecord


@dataclass(frozen=True)
class StreamConnected:
    metadata: EEGMetadata


@dataclass(frozen=True)
class PhaseChanged:
    old_phase: CalibrationPhase
    new_phase: CalibrationPhase


@dataclass(frozen=True)
class EEGWindowReady:
    chunk: EEGChunk


@dataclass(frozen=True)
class TrialStarted:
    event: BCIEvent


@dataclass(frozen=True)
class TrialCompleted:
    trial: TrialRecord


@dataclass(frozen=True)
class FeatureComputed:
    feature: FeatureRecord


@dataclass(frozen=True)
class CalibrationBatchReady:
    n_batch: int
    n_total: int


@dataclass(frozen=True)
class ModelUpdated:
    model_version: int
    metrics: dict


@dataclass(frozen=True)
class PredictionProduced:
    prediction: Prediction


@dataclass(frozen=True)
class DecisionEmitted:
    decision: Decision


@dataclass(frozen=True)
class ExperimentFinished:
    artifact_dir: str
    metrics: dict
