from __future__ import annotations

import csv
import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci.buffering.ring import TimestampedRingBuffer
from bci.config import BCIConfig, write_resolved_config
from bci.domain import CalibrationPhase, Decision, FeatureRecord, Prediction, TrialRecord
from bci.evaluation.metrics import summarize_predictions
from bci.experiment.bus import EventBus
from bci.experiment.events import (
    CalibrationBatchReady,
    CalibrationStatus,
    ChallengeRoundFinished,
    DecisionEmitted,
    EEGWindowReady,
    ExperimentFinished,
    FeatureComputed,
    FinalTestFinished,
    FitFailed,
    FitStarted,
    GroundTruthChanged,
    LiveWindowUpdated,
    ModelUpdated,
    ParameterChanged,
    PhaseChanged,
    PredictionProduced,
    ProtocolReady,
    ReplayPaused,
    ReplaySpeedChanged,
    StreamConnected,
    TrialCompleted,
    TrialStarted,
)
from bci.experiment.trial_builder import RealtimeTrialBuilder
from bci.features.base import FeatureExtractor
from bci.inference.decision import DecisionPolicy
from bci.models.base import Decoder
from bci.preprocessing.base import Preprocessor
from bci.protocol.state_machine import ProtocolAction, ProtocolCommand, ProtocolStateMachine
from bci.replay.clock import ReplayClock
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
        replay_clock: ReplayClock | None = None,
        publisher=None,
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
        self.clock = replay_clock or ReplayClock(config.source.replay.speed)
        self.publisher = publisher
        self.artifact_dir = artifact_dir or config.project.artifact_dir / utc_run_id("experiment")
        self._stop = False
        self._state_machine = ProtocolStateMachine(
            CalibrationPhase.READY if config.experiment.manual_start else CalibrationPhase.BOOTSTRAP
        )
        self._phase = self._state_machine.state
        self._calibration: list[FeatureRecord] = []
        self._validation_predictions: list[Prediction] = []
        self._test_predictions: list[Prediction] = []
        self._validation_decisions: list[Decision] = []
        self._test_decisions: list[Decision] = []
        self._all_features: list[FeatureRecord] = []
        self._history: list[dict[str, Any]] = []
        self._decisions: list[Decision] = []
        self._parameter_changes: list[dict[str, Any]] = []
        self._new_source_events_since_fit: set[int] = set()
        self._fitted_source_events: set[int] = set()
        self._actions: queue.Queue[ProtocolCommand] = queue.Queue()
        self._last_live_preview_time: float | None = None
        self._live_preview_counter = 0
        self._last_eeg_gui_publish_time: float | None = None

    def stop(self) -> None:
        self._stop = True

    def request_action(self, action: ProtocolAction | str, payload: dict | None = None) -> None:
        if isinstance(action, str):
            action = ProtocolAction(action)
        self._actions.put(ProtocolCommand(action, payload or {}))

    def run(self) -> ExperimentResult:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(self.config, self.artifact_dir / "config_resolved.yaml")
        if self.protocol_entries:
            from bci.protocol.allocation import write_protocol_manifest

            write_protocol_manifest(self.protocol_entries, self.artifact_dir / "protocol_manifest.csv")
        self.bus.publish(ProtocolReady(self._phase, self._protocol_summary()))
        if self.config.experiment.manual_start:
            self.clock.pause()
            self.bus.publish(ReplayPaused(True))
            while not self._stop and self._phase == CalibrationPhase.READY:
                self._drain_actions()
                time.sleep(self.config.experiment.poll_interval_seconds)
        if self._stop:
            metrics = {"validation": {"n": 0}, "test": {"n": 0}, "n_features": 0}
            return ExperimentResult(self.artifact_dir, metrics, 0, self.decoder.model_version)
        if self.publisher is not None:
            self.publisher.start()
        metadata = self.eeg_source.connect()
        if self.streaming_preprocessor is not None:
            self.streaming_preprocessor.reset(metadata)
        self.event_source.connect()
        self.bus.publish(StreamConnected(metadata))
        ring = TimestampedRingBuffer(self.config.source.lsl.buffer_seconds, metadata.sfreq, metadata.ch_names)
        trial_builder = RealtimeTrialBuilder(self.config, self.split_by_event)
        idle_started = time.monotonic()
        if not self.config.experiment.manual_start:
            self._set_phase(CalibrationPhase.CALIBRATION_STREAMING)
        try:
            while not self._stop:
                self._drain_actions()
                if self.clock.paused and not self.clock.consume_step():
                    time.sleep(self.config.experiment.poll_interval_seconds)
                    continue
                did_work = False
                chunk = self._poll_eeg()
                if chunk is not None:
                    did_work = True
                    if self.streaming_preprocessor is not None:
                        chunk = self.streaming_preprocessor.process_chunk(chunk)
                    ring.append(chunk)
                    if not getattr(self.eeg_source, "externally_paced", False):
                        self.clock.wait_for_chunk(chunk.data.shape[1] / chunk.sfreq)
                    idle_started = time.monotonic()
                    self._maybe_publish_eeg_window(ring)
                    if hasattr(self.event_source, "advance_to") and ring.latest_time is not None:
                        self.event_source.advance_to(ring.latest_time)  # type: ignore[attr-defined]
                    self._maybe_publish_live_preview(ring)
                events = self.event_source.poll()
                if events:
                    did_work = True
                for event in events:
                    pending = trial_builder.add_event(event)
                    if pending is not None:
                        self.bus.publish(TrialStarted(event))
                        split_key = (event.event_index or 0) * 1000
                        split = self.split_by_event.get(split_key, self.split_by_event.get(event.event_index or 0))
                        self.bus.publish(GroundTruthChanged(event.command, event.native_label, event.event_index, split))
                trials = trial_builder.resolve(ring)
                if trials:
                    did_work = True
                for trial in trials:
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
                if not did_work:
                    time.sleep(self.config.experiment.poll_interval_seconds)
        finally:
            self.event_source.close()
            self.eeg_source.close()
            if self.publisher is not None:
                self.publisher.stop()
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
            self._set_phase(CalibrationPhase.CALIBRATION_STREAMING)
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
            self._set_phase(CalibrationPhase.CHALLENGE_STREAMING)
            self._predict_and_emit(feature, self._validation_predictions, self._validation_decisions)
            return
        if feature.split == "test":
            self._set_phase(CalibrationPhase.FINAL_TEST_STREAMING)
            self._predict_and_emit(feature, self._test_predictions, self._test_decisions)
            return
        if feature.split == "reserve":
            return
        if self.decoder.model_version:
            self._set_phase(CalibrationPhase.INFERENCE)
            self._predict_and_emit(feature, self._test_predictions)

    def _maybe_publish_eeg_window(self, ring: TimestampedRingBuffer) -> None:
        if ring.latest_time is None or ring.earliest_time is None:
            return
        latest = ring.latest_time
        min_interval = 1.0 / max(self.config.gui.refresh_hz, 1.0)
        if self._last_eeg_gui_publish_time is not None and latest - self._last_eeg_gui_publish_time < min_interval:
            return
        start = max(ring.earliest_time, latest - self.config.gui.eeg_history_seconds)
        if start >= latest:
            return
        expected_samples = max(1, int(round((latest - start) * ring.sfreq)))
        try:
            window = ring.slice(start, latest, expected_samples=expected_samples)
        except ValueError:
            return
        self._last_eeg_gui_publish_time = latest
        self.bus.publish(EEGWindowReady(window))

    def _maybe_publish_live_preview(self, ring: TimestampedRingBuffer) -> None:
        if not self.config.experiment.live_preview or ring.latest_time is None:
            return
        latest = ring.latest_time
        stride = self.config.trials.inference_stride_seconds
        if self._last_live_preview_time is not None and latest - self._last_live_preview_time < stride:
            return
        start = latest - self.config.trials.window_seconds
        end = latest
        if not ring.has_interval(start, end):
            return
        expected_samples = int(round(self.config.trials.window_seconds * ring.sfreq))
        chunk = ring.slice(start, end, expected_samples=expected_samples)
        trial = TrialRecord(
            trial_id=f"live-preview-{self._live_preview_counter}",
            dataset=self.config.dataset.name,
            subject=self.config.dataset.subjects[0] if self.config.dataset.subjects else 0,
            session="live",
            run="live",
            event_index=-1,
            native_label="live_preview",
            command="NONE",
            start_time=start,
            end_time=end,
            sfreq=ring.sfreq,
            ch_names=list(ring.ch_names),
            data=chunk.data,
            source_event_id=None,
            split="preview",
        )
        self._live_preview_counter += 1
        self._last_live_preview_time = latest
        feature = self.feature_extractor.transform(self.preprocessor.transform(trial))
        prediction: Prediction | None = None
        decision: Decision | None = None
        if self.decoder.model_version:
            probs = self.decoder.predict(feature)
            label, confidence = max(probs.items(), key=lambda item: item[1])
            prediction = Prediction(
                trial_id=feature.trial_id,
                true_label=None,
                probabilities=probs,
                predicted_label=label,
                confidence=float(confidence),
                model_version=self.decoder.model_version,
                timestamp=end,
            )
            decision = self._preview_decision(prediction)
        latent_point = self._preview_latent_point(feature)
        self.bus.publish(LiveWindowUpdated(feature, prediction, decision, latent_point))

    def _preview_decision(self, prediction: Prediction) -> Decision:
        command = prediction.predicted_label
        reason = "preview_argmax"
        if prediction.confidence < self.config.decision.posterior_threshold:
            command = "NONE"
            reason = "preview_below_threshold"
        return Decision(
            timestamp=prediction.timestamp,
            command=command if self.config.decision.emit_none or command != "NONE" else "",
            probabilities=dict(prediction.probabilities),
            confidence=prediction.confidence,
            model_version=prediction.model_version,
            reason=reason,
            threshold=self.config.decision.posterior_threshold,
            consecutive=1 if command != "NONE" else 0,
            required_consecutive=1,
        )

    def _preview_latent_point(self, feature: FeatureRecord) -> list[float] | None:
        try:
            point = self.decoder.transform_latent(feature.values)[0]
        except (NotImplementedError, RuntimeError, ValueError, AttributeError):
            return None
        return [float(value) for value in point]

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
        self._set_phase(CalibrationPhase.CALIBRATION_FITTING)
        self.bus.publish(FitStarted(self.decoder.model_version + 1, self._n_unique_calibration_events()))
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
        self._set_phase(CalibrationPhase.CALIBRATION_READY)
        if self.config.experiment.manual_start:
            self.clock.pause()
            self.bus.publish(ReplayPaused(True))
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

    def _predict_and_emit(self, feature: FeatureRecord, sink: list[Prediction], decision_sink: list[Decision] | None = None) -> None:
        if self.decoder.model_version == 0:
            if not self._maybe_update_model():
                self.bus.publish(FitFailed("model unavailable; calibration requirements are not satisfied"))
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
        if decision_sink is not None:
            decision_sink.append(decision)
        self.bus.publish(DecisionEmitted(decision))
        for command_sink in self.sinks:
            command_sink.emit(decision)

    def _set_phase(self, phase: CalibrationPhase) -> None:
        if phase == self._phase:
            return
        old = self._phase
        self._phase = phase
        self._state_machine.state = phase
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
        if self._validation_predictions:
            passed = self._challenge_passed(metrics["validation"])
            self.bus.publish(ChallengeRoundFinished(metrics["validation"], passed))
            metrics["challenge_passed"] = passed
        if self._test_predictions:
            self.bus.publish(FinalTestFinished(metrics["test"]))
        if self.config.experiment.mode in {"synthetic", "classifier_smoke", "controller_smoke"}:
            metrics["smoke"] = self._smoke_summary()
        (self.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._write_features()
        self._write_predictions("predictions_validation.csv", self._validation_predictions)
        self._write_predictions("predictions_test.csv", self._test_predictions)
        self._write_predictions("challenge_predictions.csv", self._validation_predictions)
        self._write_predictions("final_test_predictions.csv", self._test_predictions)
        self._write_history()
        self._write_decisions()
        self._write_decisions_file("challenge_decisions.csv", self._validation_decisions)
        self._write_decisions_file("final_test_decisions.csv", self._test_decisions)
        self._write_parameter_changes()
        if self._test_predictions:
            (self.artifact_dir / "final_test_metrics.json").write_text(json.dumps(metrics["test"], indent=2), encoding="utf-8")
        (self.artifact_dir / "run_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._set_phase(CalibrationPhase.FINISHED)
        return metrics

    def _drain_actions(self) -> None:
        while True:
            try:
                command = self._actions.get_nowait()
            except queue.Empty:
                return
            self._handle_action(command)

    def _handle_action(self, command: ProtocolCommand) -> None:
        action = command.action
        payload = command.payload or {}
        if action == ProtocolAction.PAUSE:
            self.clock.pause()
            self.bus.publish(ReplayPaused(True))
            return
        if action == ProtocolAction.RESUME:
            self.clock.resume()
            self.bus.publish(ReplayPaused(False))
            return
        if action == ProtocolAction.STEP:
            self.clock.step()
            return
        if action == ProtocolAction.SET_SPEED:
            speed = float(payload["speed"])
            self.clock.set_speed(speed)
            self.config.source.replay.speed = speed
            self.bus.publish(ReplaySpeedChanged(speed))
            return
        if action == ProtocolAction.UPDATE_DECISION_PARAMS:
            self._update_decision_params(payload)
            return
        new_state = self._state_machine.transition(action)
        self._set_phase(new_state)
        if action == ProtocolAction.START_CALIBRATION:
            self.clock.resume()
            self.bus.publish(ReplayPaused(False))
        elif action == ProtocolAction.START_CHALLENGE:
            self.decision_policy.reset()
            self.clock.resume()
        elif action == ProtocolAction.START_FINAL_TEST:
            self.decision_policy.reset()
            self.clock.resume()

    def _update_decision_params(self, payload: dict[str, Any]) -> None:
        for key in ["posterior_threshold", "alpha", "consecutive_windows", "refractory_seconds"]:
            if key not in payload:
                continue
            old = getattr(self.config.decision, key)
            new = payload[key]
            setattr(self.config.decision, key, new)
            self._parameter_changes.append(
                {
                    "timestamp": time.time(),
                    "phase": self._phase.value,
                    "parameter": f"decision.{key}",
                    "old_value": old,
                    "new_value": new,
                    "requires_refit": False,
                    "model_version_before": self.decoder.model_version,
                    "model_version_after": self.decoder.model_version,
                }
            )
            self.bus.publish(ParameterChanged(f"decision.{key}", old, new, False, self.decoder.model_version))

    def _protocol_summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for entry in self.protocol_entries:
            key = f"{entry.role}:{entry.command}"
            counts[key] = counts.get(key, 0) + 1
        return {"n_events": len(self.protocol_entries), "counts": counts}

    def _challenge_passed(self, metrics: dict[str, Any]) -> bool:
        if not metrics or metrics.get("n", 0) < self.config.protocol.challenge.minimum_events:
            return False
        if metrics.get(self.config.protocol.challenge.metric, 0.0) < self.config.protocol.challenge.pass_threshold:
            return False
        return metrics.get("false_commands_per_minute_rest", 0.0) <= self.config.protocol.challenge.max_false_commands_per_minute

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
        self._write_decisions_file("decisions.csv", self._decisions)

    def _write_decisions_file(self, name: str, decisions: list[Decision]) -> None:
        with (self.artifact_dir / name).open("w", newline="", encoding="utf-8") as f:
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
            for d in decisions:
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

    def _write_parameter_changes(self) -> None:
        with (self.artifact_dir / "parameter_change_log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "phase",
                    "parameter",
                    "old_value",
                    "new_value",
                    "requires_refit",
                    "model_version_before",
                    "model_version_after",
                ],
            )
            writer.writeheader()
            writer.writerows(self._parameter_changes)
