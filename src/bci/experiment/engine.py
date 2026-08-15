from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci.buffering.ring import TimestampedRingBuffer
from bci.config import BCIConfig, write_resolved_config
from bci.domain import CalibrationPhase, Decision, FeatureRecord, Prediction
from bci.evaluation.metrics import summarize_predictions
from bci.experiment.bus import EventBus
from bci.experiment.events import (
    CalibrationBatchReady,
    CalibrationStatus,
    DecisionEmitted,
    EEGWindowReady,
    ExperimentFinished,
    FeatureComputed,
    ModelUpdated,
    PhaseChanged,
    PredictionProduced,
    StreamConnected,
    TrialCompleted,
    TrialStarted,
)
from bci.experiment.trial_builder import RealtimeTrialBuilder
from bci.features.base import FeatureExtractor
from bci.inference.decision import DecisionPolicy
from bci.models.base import Decoder
from bci.preprocessing.base import Preprocessor
from bci.sinks.base import CommandSink
from bci.sources.base import EEGSource, EventSource
from bci.utils.timing import utc_run_id


@dataclass(frozen=True)
class ExperimentResult:
    artifact_dir: Path
    metrics: dict[str, Any]
    n_trials: int
    model_version: int


class RealtimeExperimentEngine:
    def __init__(
        self,
        config: BCIConfig,
        eeg_source: EEGSource,
        event_source: EventSource,
        preprocessor: Preprocessor,
        feature_extractor: FeatureExtractor,
        decoder: Decoder,
        decision_policy: DecisionPolicy,
        sinks: list[CommandSink],
        event_bus: EventBus,
        split_by_event: dict[int, str],
        streaming_preprocessor=None,
        protocol_entries=None,
        artifact_dir: Path | None = None,
    ):
        self.config = config
        self.eeg_source = eeg_source
        self.event_source = event_source
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.decision_policy = decision_policy
        self.sinks = sinks
        self.bus = event_bus
        self.split_by_event = split_by_event
        self.streaming_preprocessor = streaming_preprocessor
        self.protocol_entries = protocol_entries or []
        self.artifact_dir = artifact_dir or config.project.artifact_dir / utc_run_id("experiment")
        self._stop = False
        self._phase = CalibrationPhase.BOOTSTRAP
        self._calibration: list[FeatureRecord] = []
        self._validation_predictions: list[Prediction] = []
        self._test_predictions: list[Prediction] = []
        self._all_features: list[FeatureRecord] = []
        self._history: list[dict[str, Any]] = []
        self._decisions: list[Decision] = []
        self._new_source_events_since_fit: set[int] = set()
        self._fitted_source_events: set[int] = set()

    def stop(self) -> None:
        self._stop = True

    def run(self) -> ExperimentResult:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(self.config, self.artifact_dir / "config_resolved.yaml")
        if self.protocol_entries:
            from bci.protocol.allocation import write_protocol_manifest

            write_protocol_manifest(self.protocol_entries, self.artifact_dir / "protocol_manifest.csv")
        metadata = self.eeg_source.connect()
        if self.streaming_preprocessor is not None:
            self.streaming_preprocessor.reset(metadata)
        self.event_source.connect()
        self.bus.publish(StreamConnected(metadata))
        ring = TimestampedRingBuffer(self.config.source.lsl.buffer_seconds, metadata.sfreq, metadata.ch_names)
        trial_builder = RealtimeTrialBuilder(self.config, self.split_by_event)
        idle_started = time.monotonic()
        self._set_phase(CalibrationPhase.CALIBRATING)
        try:
            while not self._stop:
                chunk = self._poll_eeg()
                if chunk is not None:
                    if self.streaming_preprocessor is not None:
                        chunk = self.streaming_preprocessor.process_chunk(chunk)
                    ring.append(chunk)
                    idle_started = time.monotonic()
                    self.bus.publish(EEGWindowReady(chunk))
                    if hasattr(self.event_source, "advance_to") and ring.latest_time is not None:
                        self.event_source.advance_to(ring.latest_time)  # type: ignore[attr-defined]
                for event in self.event_source.poll():
                    pending = trial_builder.add_event(event)
                    if pending is not None:
                        self.bus.publish(TrialStarted(event))
                for trial in trial_builder.resolve(ring):
                    self.bus.publish(TrialCompleted(trial))
                    feature = self.feature_extractor.transform(self.preprocessor.transform(trial))
                    self._all_features.append(feature)
                    self.bus.publish(FeatureComputed(feature))
                    self._route_feature(feature)
                    if self.config.experiment.max_trials and len(self._all_features) >= self.config.experiment.max_trials:
                        self._stop = True
                if self._is_complete():
                    self._stop = True
                if time.monotonic() - idle_started > self.config.experiment.max_idle_seconds:
                    self._stop = True
                time.sleep(self.config.experiment.poll_interval_seconds)
        finally:
            self.event_source.close()
            self.eeg_source.close()
        metrics = self._finalize()
        result = ExperimentResult(
            artifact_dir=self.artifact_dir,
            metrics=metrics,
            n_trials=len(self._all_features),
            model_version=self.decoder.model_version,
        )
        self.bus.publish(ExperimentFinished(str(self.artifact_dir), metrics))
        return result

    def _poll_eeg(self):
        if hasattr(self.eeg_source, "poll_new"):
            return self.eeg_source.poll_new()  # type: ignore[attr-defined]
        return self.eeg_source.read_latest(self.config.experiment.poll_interval_seconds)

    def _route_feature(self, feature: FeatureRecord) -> None:
        if feature.split == "calibration":
            self._set_phase(CalibrationPhase.CALIBRATING)
            self._calibration.append(feature)
            source_event_id = int(feature.provenance.get("source_event_id", feature.provenance["event_index"]))
            if source_event_id not in self._fitted_source_events:
                self._new_source_events_since_fit.add(source_event_id)
            fit_every = self.config.protocol.fit_every_new_events or self.config.calibration.batch_size_trials
            if len(self._new_source_events_since_fit) >= fit_every:
                self.bus.publish(CalibrationBatchReady(len(self._new_source_events_since_fit), self._n_unique_calibration_events()))
                self._maybe_update_model()
            self.bus.publish(self._calibration_status())
            return
        if feature.split == "validation":
            self._set_phase(CalibrationPhase.VALIDATING)
            self._predict_and_emit(feature, self._validation_predictions)
            return
        if feature.split == "test":
            self._set_phase(CalibrationPhase.FROZEN_TEST)
            self._predict_and_emit(feature, self._test_predictions)
            return
        if feature.split == "reserve":
            return
        if self.decoder.model_version:
            self._set_phase(CalibrationPhase.INFERENCE)
            self._predict_and_emit(feature, self._test_predictions)

    def _maybe_update_model(self) -> bool:
        expected = list(self.config.protocol.classes)
        counts = self._calibration_event_counts()
        if any(counts.get(label, 0) < self.config.protocol.minimum_events_per_class_before_fit for label in expected):
            return False
        if self._n_unique_calibration_events() < self.config.calibration.batch_size_trials:
            return False
        if len(self._new_source_events_since_fit) < (self.config.protocol.fit_every_new_events or self.config.calibration.batch_size_trials):
            return False
        if not self.config.calibration.refit_on_all_accumulated_data:
            raise NotImplementedError(
                "refit_on_all_accumulated_data=false requires a decoder with true incremental updates; "
                f"{type(self.decoder).__name__} currently supports cumulative refitting only"
            )
        self.decoder.update(self._calibration)
        try:
            diagnostics = self.decoder.diagnostics(self._calibration)
        except NotImplementedError:
            diagnostics = None
        metrics = {
            "n_calibration": len(self._calibration),
            "n_original_events": self._n_unique_calibration_events(),
            "separation": diagnostics.separation if diagnostics is not None else {},
        }
        self._history.append(
            {
                "model_version": self.decoder.model_version,
                "n_records": len(self._calibration),
                "record_ids": ";".join(r.trial_id for r in self._calibration),
                "n_original_events": self._n_unique_calibration_events(),
                "separation": json.dumps(metrics["separation"], sort_keys=True),
            }
        )
        self.decoder.save(self.artifact_dir / f"model_v{self.decoder.model_version:03d}.pkl")
        self._fitted_source_events = {
            int(r.provenance.get("source_event_id", r.provenance["event_index"])) for r in self._calibration
        }
        self._new_source_events_since_fit.clear()
        self.decision_policy.reset()
        self.bus.publish(ModelUpdated(self.decoder.model_version, metrics, diagnostics))
        return True

    def _calibration_status(self) -> CalibrationStatus:
        required = self.config.protocol.minimum_events_per_class_before_fit
        counts = {label: self._calibration_event_counts().get(label, 0) for label in self.config.protocol.classes}
        missing = {label: max(0, required - count) for label, count in counts.items()}
        ready = (
            all(value == 0 for value in missing.values())
            and self._n_unique_calibration_events() >= self.config.calibration.batch_size_trials
            and len(self._new_source_events_since_fit) >= self.config.protocol.fit_every_new_events
        )
        if self.decoder.model_version > 0 and not self._new_source_events_since_fit:
            reason = f"MODEL v{self.decoder.model_version} TRAINED on {self._n_unique_calibration_events()} original events"
        elif any(missing.values()):
            needed = ", ".join(f"{label} +{value}" for label, value in missing.items() if value)
            reason = f"collecting calibration examples: waiting for {needed}"
        elif self._n_unique_calibration_events() < self.config.calibration.batch_size_trials:
            reason = f"waiting for initial batch {self._n_unique_calibration_events()}/{self.config.calibration.batch_size_trials} original events"
        elif len(self._new_source_events_since_fit) < self.config.protocol.fit_every_new_events:
            reason = f"waiting for new events {len(self._new_source_events_since_fit)}/{self.config.protocol.fit_every_new_events}"
        else:
            reason = "ready for next calibration update"
        return CalibrationStatus(
            counts=counts,
            required_per_class=required,
            batch_size=self.config.calibration.batch_size_trials,
            n_total=self._n_unique_calibration_events(),
            model_version=self.decoder.model_version,
            ready_to_fit=ready,
            reason=reason,
        )

    def _calibration_event_counts(self) -> dict[str, int]:
        seen: set[tuple[str, int]] = set()
        counts: dict[str, int] = {}
        for record in self._calibration:
            source_event_id = int(record.provenance.get("source_event_id", record.provenance["event_index"]))
            key = (record.label, source_event_id)
            if key in seen:
                continue
            seen.add(key)
            counts[record.label] = counts.get(record.label, 0) + 1
        return counts

    def _n_unique_calibration_events(self) -> int:
        return len({int(r.provenance.get("source_event_id", r.provenance["event_index"])) for r in self._calibration})

    def _predict_and_emit(self, feature: FeatureRecord, sink: list[Prediction]) -> None:
        if self.decoder.model_version == 0:
            self._maybe_update_model()
        if self.decoder.model_version == 0:
            return
        probs = self.decoder.predict(feature)
        label, confidence = max(probs.items(), key=lambda item: item[1])
        prediction = Prediction(
            trial_id=feature.trial_id,
            true_label=feature.label,
            probabilities=probs,
            predicted_label=label,
            confidence=float(confidence),
            model_version=self.decoder.model_version,
            timestamp=float(feature.provenance.get("end_time", time.time())),
        )
        sink.append(prediction)
        self.bus.publish(PredictionProduced(prediction))
        decision = self.decision_policy.update(prediction)
        self._decisions.append(decision)
        self.bus.publish(DecisionEmitted(decision))
        for command_sink in self.sinks:
            command_sink.emit(decision)

    def _set_phase(self, phase: CalibrationPhase) -> None:
        if phase == self._phase:
            return
        old = self._phase
        self._phase = phase
        self.bus.publish(PhaseChanged(old, phase))

    def _is_complete(self) -> bool:
        expected = len(self.split_by_event)
        return expected > 0 and len(self._all_features) >= expected and not any(f.split == "inference" for f in self._all_features)

    def _finalize(self) -> dict[str, Any]:
        if self.decoder.model_version == 0 and self._calibration:
            self._maybe_update_model()
        classes = list(self.decoder.classes_) if self.decoder.model_version else ["LEFT", "RIGHT", "NONE"]
        metrics = {
            "validation": summarize_predictions(self._validation_predictions, classes) if self._validation_predictions else {"n": 0},
            "test": summarize_predictions(self._test_predictions, classes) if self._test_predictions else {"n": 0},
            "n_features": len(self._all_features),
        }
        if self.config.experiment.mode in {"synthetic", "classifier_smoke", "controller_smoke"}:
            metrics["smoke"] = self._smoke_summary()
        (self.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._write_features()
        self._write_predictions("predictions_validation.csv", self._validation_predictions)
        self._write_predictions("predictions_test.csv", self._test_predictions)
        self._write_history()
        self._write_decisions()
        return metrics

    def _smoke_summary(self) -> dict[str, Any]:
        emitted = [d.command for d in self._decisions if d.command and d.command != "NONE"]
        reasons: dict[str, int] = {}
        for decision in self._decisions:
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
        if self.config.experiment.mode == "controller_smoke":
            purpose = "tests posterior smoothing, consecutive-window decisions, and command latency mechanics"
            expected = "LEFT and RIGHT commands should be emitted after repeated windows once the model is trained"
        else:
            purpose = "tests event -> window -> FFT/features -> Gaussian latent decoder with one prediction per trial"
            expected = "near-perfect validation/test classification; decisions use alpha=1.0 and consecutive_windows=1"
        return {
            "mode": self.config.experiment.mode,
            "difficulty": self.config.experiment.synthetic_difficulty,
            "purpose": purpose,
            "expected": expected,
            "model_version": self.decoder.model_version,
            "n_calibration_features": len(self._calibration),
            "n_predictions": len(self._validation_predictions) + len(self._test_predictions),
            "emitted_commands": {command: emitted.count(command) for command in sorted(set(emitted))},
            "decision_reasons": reasons,
            "decision_policy": {
                "threshold": self.config.decision.posterior_threshold,
                "consecutive_windows": self.config.decision.consecutive_windows,
                "alpha": self.config.decision.alpha,
            },
        }

    def _write_features(self) -> None:
        path = self.artifact_dir / "features.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["trial_id", "label", "split", "values", "frequency_scores", "provenance"])
            writer.writeheader()
            for record in self._all_features:
                writer.writerow(
                    {
                        "trial_id": record.trial_id,
                        "label": record.label,
                        "split": record.split,
                        "values": json.dumps(record.values.tolist()),
                        "frequency_scores": json.dumps(record.frequency_scores, sort_keys=True),
                        "provenance": json.dumps(record.provenance, sort_keys=True),
                    }
                )

    def _write_predictions(self, name: str, predictions: list[Prediction]) -> None:
        path = self.artifact_dir / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["trial_id", "true_label", "predicted_label", "confidence", "model_version", "probabilities"])
            writer.writeheader()
            for p in predictions:
                writer.writerow(
                    {
                        "trial_id": p.trial_id,
                        "true_label": p.true_label,
                        "predicted_label": p.predicted_label,
                        "confidence": p.confidence,
                        "model_version": p.model_version,
                        "probabilities": json.dumps(p.probabilities, sort_keys=True),
                    }
                )

    def _write_history(self) -> None:
        with (self.artifact_dir / "calibration_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model_version", "n_records", "n_original_events", "record_ids", "separation"])
            writer.writeheader()
            writer.writerows(self._history)

    def _write_decisions(self) -> None:
        with (self.artifact_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "command",
                    "confidence",
                    "model_version",
                    "reason",
                    "threshold",
                    "consecutive",
                    "required_consecutive",
                    "probabilities",
                ],
            )
            writer.writeheader()
            for d in self._decisions:
                writer.writerow(
                    {
                        "timestamp": d.timestamp,
                        "command": d.command,
                        "confidence": d.confidence,
                        "model_version": d.model_version,
                        "reason": d.reason,
                        "threshold": d.threshold,
                        "consecutive": d.consecutive,
                        "required_consecutive": d.required_consecutive,
                        "probabilities": json.dumps(d.probabilities, sort_keys=True),
                    }
                )
