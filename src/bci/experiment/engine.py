from __future__ import annotations

import csv
import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci.adaptation.riemannian import RiemannianPrototypeAdaptor
from bci.buffering.ring import TimestampedRingBuffer
from bci.config import BCIConfig, write_resolved_config
from bci.domain import BCIEvent, CalibrationPhase, Decision, FeatureRecord, OnlineObservation, Prediction, TrialRecord
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
    InferenceUpdated,
    LiveWindowUpdated,
    ModelUpdated,
    OnlineInferenceProduced,
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
from bci.features.alignment import EuclideanAlignment
from bci.features.base import FeatureExtractor
from bci.inference.decision import DecisionPolicy
from bci.inference.quality import quality_adjust_prediction
from bci.models.base import Decoder
from bci.preprocessing.base import Preprocessor
from bci.preprocessing.quality import SignalQualityEstimator
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
        self._online_observations: list[OnlineObservation] = []
        self._history: list[dict[str, Any]] = []
        self._decisions: list[Decision] = []
        self._parameter_changes: list[dict[str, Any]] = []
        self._adaptation_log: list[dict[str, Any]] = []
        self._seen_events: list[BCIEvent] = []
        self._new_source_events_since_fit: set[int] = set()
        self._fitted_source_events: set[int] = set()
        self._actions: queue.Queue[ProtocolCommand] = queue.Queue()
        self._last_online_inference_time: float | None = None
        self._online_inference_counter = 0
        self._last_eeg_gui_publish_time: float | None = None
        self._manual_calibration_seconds = {label: 0.0 for label in config.protocol.classes}
        self.quality_estimator = SignalQualityEstimator(config) if config.quality.enabled else None
        self.aligner = EuclideanAlignment(config)
        self.adaptor = RiemannianPrototypeAdaptor(config)

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
        if self.quality_estimator is not None:
            self.quality_estimator.reset()
        self.aligner.reset()
        self.adaptor.reset()
        self.event_source.connect()
        self.bus.publish(StreamConnected(metadata))
        ring = TimestampedRingBuffer(self.config.source.lsl.buffer_seconds, metadata.sfreq, metadata.ch_names)
        raw_ring = TimestampedRingBuffer(self.config.source.lsl.buffer_seconds, metadata.sfreq, metadata.ch_names)
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
                    raw_chunk = chunk
                    raw_ring.append(raw_chunk)
                    if self.streaming_preprocessor is not None:
                        chunk = self.streaming_preprocessor.process_chunk(chunk)
                    ring.append(chunk)
                    if not getattr(self.eeg_source, "externally_paced", False):
                        self.clock.wait_for_chunk(raw_chunk.data.shape[1] / raw_chunk.sfreq)
                    idle_started = time.monotonic()
                    self._maybe_publish_eeg_window(raw_ring)
                    if hasattr(self.event_source, "advance_to") and ring.latest_time is not None:
                        self.event_source.advance_to(ring.latest_time)  # type: ignore[attr-defined]
                events = self.event_source.poll()
                if events:
                    did_work = True
                    self._seen_events.extend(events)
                for event in events:
                    if self.config.experiment.manual_start and self._phase == CalibrationPhase.CALIBRATION_STREAMING:
                        pending = self._add_manual_calibration_event(trial_builder, event)
                    else:
                        allowed_splits = self._allowed_splits_for_current_phase()
                        pending = trial_builder.add_event(event, allowed_splits=allowed_splits)
                    if pending is not None:
                        self.bus.publish(TrialStarted(event))
                        split = pending.split
                        self.bus.publish(GroundTruthChanged(event.command, event.native_label, event.event_index, split))
                trials = trial_builder.resolve(ring)
                if trials:
                    did_work = True
                for trial in trials:
                    if not self._trial_allowed_for_current_phase(trial):
                        continue
                    self.bus.publish(TrialCompleted(trial))
                    feature = self.feature_extractor.transform(self.preprocessor.transform(trial))
                    feature = self.aligner.update_transform(feature)
                    if not self._feature_allowed_for_current_phase(feature):
                        continue
                    self._all_features.append(feature)
                    self.bus.publish(FeatureComputed(feature))
                    self._route_feature(feature)
                    if self.config.experiment.max_trials and len(self._all_features) >= self.config.experiment.max_trials:
                        self._stop = True
                if chunk is not None:
                    self._maybe_run_online_inference(ring)
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

    def _add_manual_calibration_event(self, trial_builder: RealtimeTrialBuilder, event) -> object | None:
        if event.command is None or event.command not in self.config.protocol.classes:
            return None
        remaining = self.config.calibration.seconds_per_class - self._manual_calibration_seconds.get(event.command, 0.0)
        if remaining <= 1e-9:
            return None
        usable = self._usable_calibration_duration(event, remaining)
        pending = trial_builder.add_event(
            event,
            allowed_splits={"calibration"},
            split_override="calibration",
            max_duration_seconds=usable,
        )
        if pending is not None:
            self._manual_calibration_seconds[event.command] = self._manual_calibration_seconds.get(event.command, 0.0) + usable
        return pending

    def _usable_calibration_duration(self, event, remaining: float) -> float:
        if event.duration > self.config.trials.onset_offset_seconds:
            available = event.duration - self.config.trials.onset_offset_seconds
        elif event.duration > 0:
            available = event.duration
        else:
            available = self.config.trials.window_seconds
        return max(0.0, min(remaining, available))

    def _route_feature(self, feature: FeatureRecord) -> None:
        if feature.split == "calibration":
            if not self.config.experiment.manual_start:
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
            if not self.config.experiment.manual_start:
                self._set_phase(CalibrationPhase.CHALLENGE_STREAMING)
            if self.config.experiment.online_inference:
                self._predict_for_evaluation(feature, self._validation_predictions)
            else:
                self._predict_and_emit(feature, self._validation_predictions, self._validation_decisions)
            return
        if feature.split == "test":
            if not self.config.experiment.manual_start:
                self._set_phase(CalibrationPhase.FINAL_TEST_STREAMING)
            if self.config.experiment.online_inference:
                self._predict_for_evaluation(feature, self._test_predictions)
            else:
                self._predict_and_emit(feature, self._test_predictions, self._test_decisions)
            return
        if feature.split == "reserve":
            return
        if self.decoder.model_version:
            self._set_phase(CalibrationPhase.INFERENCE)
            self._predict_and_emit(feature, self._test_predictions)

    def _allowed_splits_for_current_phase(self) -> set[str] | None:
        if not self.config.experiment.manual_start:
            return None
        if self._phase == CalibrationPhase.CALIBRATION_STREAMING:
            return {"calibration"}
        if self._phase == CalibrationPhase.APPEND_CALIBRATION:
            return {"reserve"}
        if self._phase == CalibrationPhase.CHALLENGE_STREAMING:
            return {"validation"}
        if self._phase == CalibrationPhase.FINAL_TEST_STREAMING:
            return {"test"}
        return set()

    def _trial_allowed_for_current_phase(self, trial: TrialRecord) -> bool:
        allowed = self._allowed_splits_for_current_phase()
        return allowed is None or trial.split in allowed

    def _feature_allowed_for_current_phase(self, feature: FeatureRecord) -> bool:
        allowed = self._allowed_splits_for_current_phase()
        return allowed is None or feature.split in allowed

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

    def _maybe_run_online_inference(self, ring: TimestampedRingBuffer) -> None:
        total_started = time.perf_counter()
        if not (self.config.experiment.live_preview or self.config.experiment.online_inference) or ring.latest_time is None:
            return
        latest = ring.latest_time
        stride = self.config.experiment.online_inference_stride_seconds or self.config.trials.inference_stride_seconds
        if self._last_online_inference_time is not None and latest - self._last_online_inference_time < stride:
            return
        start = latest - self.config.trials.window_seconds
        end = latest
        if not ring.has_interval(start, end):
            return
        expected_samples = int(round(self.config.trials.window_seconds * ring.sfreq))
        latency_ms: dict[str, float] = {}
        materialize_started = time.perf_counter()
        chunk = ring.slice(start, end, expected_samples=expected_samples)
        trial = TrialRecord(
            trial_id=f"online-{self._online_inference_counter:06d}",
            dataset=self.config.dataset.name,
            subject=self.config.dataset.subjects[0] if self.config.dataset.subjects else 0,
            session="live",
            run="live",
            event_index=-1,
            native_label="online",
            command=self._ground_truth_for_window(start, end) or "NONE",
            start_time=start,
            end_time=end,
            sfreq=ring.sfreq,
            ch_names=list(ring.ch_names),
            data=chunk.data,
            source_event_id=None,
            split="online",
        )
        latency_ms["window_materialization"] = self._elapsed_ms(materialize_started)
        self._online_inference_counter += 1
        self._last_online_inference_time = latest
        feature_started = time.perf_counter()
        feature = self.feature_extractor.transform(self.preprocessor.transform(trial))
        latency_ms["feature_extraction"] = self._elapsed_ms(feature_started)
        quality_started = time.perf_counter()
        quality = self.quality_estimator.estimate(chunk.data, ring.sfreq, list(ring.ch_names)) if self.quality_estimator is not None else None
        latency_ms["quality"] = self._elapsed_ms(quality_started)
        alignment_started = time.perf_counter()
        feature = self.aligner.update_transform(feature, quality)
        latency_ms["alignment"] = self._elapsed_ms(alignment_started)
        prediction: Prediction | None = None
        evidence_prediction: Prediction | None = None
        decision: Decision | None = None
        quality_action = "not_evaluated"
        emitted = False
        ground_truth = self._ground_truth_for_window(start, end)
        if self.decoder.model_version:
            decoder_started = time.perf_counter()
            prediction = self._make_prediction(feature, true_label=ground_truth)
            evidence_prediction, quality_action = quality_adjust_prediction(
                prediction,
                quality,
                self.config.quality.hard_reject_threshold,
            )
            latency_ms["decoder"] = self._elapsed_ms(decoder_started)
            if self.config.experiment.online_inference and self._online_control_allowed():
                decision_started = time.perf_counter()
                decision = self.decision_policy.update(evidence_prediction)
                self._decisions.append(decision)
                decision_sink = self._online_decision_sink_for_phase()
                if decision_sink is not None:
                    decision_sink.append(decision)
                self.bus.publish(DecisionEmitted(decision))
                for command_sink in self.sinks:
                    command_sink.emit(decision)
                emitted = True
                latency_ms["decision"] = self._elapsed_ms(decision_started)
        latency_ms.setdefault("decoder", 0.0)
        latency_ms.setdefault("decision", 0.0)
        latent_started = time.perf_counter()
        latent_point = self._latent_point(feature)
        latency_ms["latent"] = self._elapsed_ms(latent_started)
        observation = OnlineObservation(
            window_id=feature.trial_id,
            window_start=start,
            window_end=end,
            phase=self._phase,
            feature=feature,
            prediction=prediction,
            evidence_prediction=evidence_prediction,
            decision=decision,
            quality=quality,
            quality_action=quality_action,
            latency_ms=latency_ms,
            current_ground_truth_if_known=ground_truth,
            model_version=self.decoder.model_version,
            alignment_version=feature.alignment_version,
            emitted=emitted,
        )
        self._online_observations.append(observation)
        publish_started = time.perf_counter()
        self.bus.publish(OnlineInferenceProduced(observation, latent_point))
        self.bus.publish(LiveWindowUpdated(feature, prediction, decision, latent_point))
        latency_ms["publication"] = self._elapsed_ms(publish_started)
        adaptation_started = time.perf_counter()
        adaptation_row = self.adaptor.update(observation, self.decoder)
        latency_ms["adaptation"] = self._elapsed_ms(adaptation_started)
        latency_ms["total_compute"] = self._elapsed_ms(total_started)
        if adaptation_row["reason"] != "disabled":
            self._adaptation_log.append(adaptation_row)

    def _elapsed_ms(self, started: float) -> float:
        return (time.perf_counter() - started) * 1000.0

    def _online_control_allowed(self) -> bool:
        if self._phase in {
            CalibrationPhase.CHALLENGE_STREAMING,
            CalibrationPhase.FINAL_TEST_STREAMING,
            CalibrationPhase.INFERENCE,
        }:
            return True
        if self.config.source.mode == "lsl_live" and self.decoder.model_version:
            self._set_phase(CalibrationPhase.INFERENCE)
            return True
        return False

    def _online_decision_sink_for_phase(self) -> list[Decision] | None:
        if self._phase == CalibrationPhase.CHALLENGE_STREAMING:
            return self._validation_decisions
        if self._phase in {CalibrationPhase.FINAL_TEST_STREAMING, CalibrationPhase.FROZEN_TEST}:
            return self._test_decisions
        return None

    def _ground_truth_for_window(self, start: float, end: float) -> str | None:
        midpoint = (start + end) / 2.0
        best: BCIEvent | None = None
        for event in self._seen_events:
            event_end = event.timestamp + max(event.duration, 0.0)
            if event.timestamp <= midpoint <= event_end and event.command is not None:
                if best is None or event.timestamp >= best.timestamp:
                    best = event
        return best.command if best is not None else None

    def _latent_point(self, feature: FeatureRecord) -> list[float] | None:
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
        if self.config.experiment.manual_start and not self._manual_calibration_complete():
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
        elif self.config.experiment.manual_start and not self._manual_calibration_complete():
            needed = ", ".join(
                f"{label} {min(seconds, self.config.calibration.seconds_per_class):.1f}/{self.config.calibration.seconds_per_class:g}s"
                for label, seconds in self._manual_calibration_seconds.items()
            )
            reason = f"collecting calibration seconds: {needed}"
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

    def _manual_calibration_complete(self) -> bool:
        target = self.config.calibration.seconds_per_class
        return all(self._manual_calibration_seconds.get(label, 0.0) >= target for label in self.config.protocol.classes)

    def _n_unique_calibration_events(self) -> int:
        return len({int(r.provenance.get("source_event_id", r.provenance["event_index"])) for r in self._calibration})

    def _ensure_model_available(self) -> bool:
        if self.decoder.model_version == 0:
            if not self._maybe_update_model():
                self.bus.publish(FitFailed("model unavailable; calibration requirements are not satisfied"))
        return self.decoder.model_version > 0

    def _make_prediction(self, feature: FeatureRecord, true_label: str | None) -> Prediction:
        probs = self.decoder.predict(feature)
        label, confidence = max(probs.items(), key=lambda item: item[1])
        return Prediction(
            trial_id=feature.trial_id,
            true_label=true_label,
            probabilities=probs,
            predicted_label=label,
            confidence=float(confidence),
            model_version=self.decoder.model_version,
            timestamp=float(feature.provenance.get("end_time", time.time())),
        )

    def _predict_for_evaluation(self, feature: FeatureRecord, sink: list[Prediction]) -> None:
        if not self._ensure_model_available():
            return
        prediction = self._make_prediction(feature, true_label=feature.label)
        sink.append(prediction)
        self.bus.publish(PredictionProduced(prediction))
        self.bus.publish(InferenceUpdated(feature, prediction, None, self._latent_point(feature)))

    def _predict_and_emit(self, feature: FeatureRecord, sink: list[Prediction], decision_sink: list[Decision] | None = None) -> None:
        if not self._ensure_model_available():
            return
        prediction = self._make_prediction(feature, true_label=feature.label)
        sink.append(prediction)
        self.bus.publish(PredictionProduced(prediction))
        decision = self.decision_policy.update(prediction)
        self._decisions.append(decision)
        if decision_sink is not None:
            decision_sink.append(decision)
        self.bus.publish(DecisionEmitted(decision))
        self.bus.publish(InferenceUpdated(feature, prediction, decision, self._latent_point(feature)))
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
        if self.config.experiment.manual_start:
            return False
        expected = set(self.split_by_event)
        if not expected:
            return False
        seen: set[int] = set()
        for feature in self._all_features:
            event_index = int(feature.provenance["event_index"])
            source_event_id = int(feature.provenance.get("source_event_id", event_index))
            if event_index in expected:
                seen.add(event_index)
            elif source_event_id in expected:
                seen.add(source_event_id)
        return expected.issubset(seen) and not any(f.split == "inference" for f in self._all_features)

    def _finalize(self) -> dict[str, Any]:
        if self.decoder.model_version == 0 and self._calibration:
            self._maybe_update_model()
        classes = list(self.decoder.classes_) if self.decoder.model_version else ["LEFT", "RIGHT", "NONE"]
        metrics = {
            "validation": summarize_predictions(self._validation_predictions, classes) if self._validation_predictions else {"n": 0},
            "test": summarize_predictions(self._test_predictions, classes) if self._test_predictions else {"n": 0},
            "n_features": len(self._all_features),
            "n_online_observations": len(self._online_observations),
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
        self._write_online_observations()
        self._write_alignment_status()
        self._write_adaptation_log()
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
        if action in {ProtocolAction.START_CHALLENGE, ProtocolAction.START_FINAL_TEST} and self.decoder.model_version == 0:
            self.bus.publish(FitFailed("cannot start inference phase before a model is trained"))
            return
        new_state = self._state_machine.transition(action)
        self._set_phase(new_state)
        if action == ProtocolAction.START_CALIBRATION:
            self._manual_calibration_seconds = {label: 0.0 for label in self.config.protocol.classes}
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

    def _write_online_observations(self) -> None:
        csv_path = self.artifact_dir / "online_observations.csv"
        jsonl_path = self.artifact_dir / "online_observations.jsonl"
        fieldnames = [
            "window_id",
            "window_start",
            "window_end",
            "phase",
            "ground_truth",
            "predicted_label",
            "prediction_confidence",
            "evidence_predicted_label",
            "evidence_confidence",
            "decision_command",
            "decision_reason",
            "decision_confidence",
            "model_version",
            "alignment_version",
            "emitted",
            "probabilities",
            "evidence_probabilities",
            "quality_score",
            "quality_flags",
            "quality_action",
            "quality_history_ready",
            "latency_window_materialization_ms",
            "latency_feature_extraction_ms",
            "latency_quality_ms",
            "latency_alignment_ms",
            "latency_decoder_ms",
            "latency_decision_ms",
            "latency_latent_ms",
            "latency_publication_ms",
            "latency_adaptation_ms",
            "latency_total_compute_ms",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for observation in self._online_observations:
                writer.writerow(self._online_observation_row(observation))
        with jsonl_path.open("w", encoding="utf-8") as f:
            for observation in self._online_observations:
                row = self._online_observation_row(observation)
                row["feature_values"] = observation.feature.values.tolist()
                row["frequency_scores"] = observation.feature.frequency_scores
                f.write(json.dumps(row, sort_keys=True) + "\n")

    def _online_observation_row(self, observation: OnlineObservation) -> dict[str, Any]:
        prediction = observation.prediction
        evidence_prediction = observation.evidence_prediction
        decision = observation.decision
        return {
            "window_id": observation.window_id,
            "window_start": observation.window_start,
            "window_end": observation.window_end,
            "phase": observation.phase.value,
            "ground_truth": observation.current_ground_truth_if_known,
            "predicted_label": prediction.predicted_label if prediction is not None else None,
            "prediction_confidence": prediction.confidence if prediction is not None else None,
            "evidence_predicted_label": evidence_prediction.predicted_label if evidence_prediction is not None else None,
            "evidence_confidence": evidence_prediction.confidence if evidence_prediction is not None else None,
            "decision_command": decision.command if decision is not None else None,
            "decision_reason": decision.reason if decision is not None else None,
            "decision_confidence": decision.confidence if decision is not None else None,
            "model_version": observation.model_version,
            "alignment_version": observation.alignment_version,
            "emitted": observation.emitted,
            "probabilities": json.dumps(prediction.probabilities, sort_keys=True) if prediction is not None else "{}",
            "evidence_probabilities": json.dumps(evidence_prediction.probabilities, sort_keys=True)
            if evidence_prediction is not None
            else "{}",
            "quality_score": observation.quality.score if observation.quality is not None else None,
            "quality_flags": ";".join(observation.quality.flags) if observation.quality is not None else "",
            "quality_action": observation.quality_action,
            "quality_history_ready": observation.quality.history_ready if observation.quality is not None else False,
            "latency_window_materialization_ms": observation.latency_ms.get("window_materialization", 0.0),
            "latency_feature_extraction_ms": observation.latency_ms.get("feature_extraction", 0.0),
            "latency_quality_ms": observation.latency_ms.get("quality", 0.0),
            "latency_alignment_ms": observation.latency_ms.get("alignment", 0.0),
            "latency_decoder_ms": observation.latency_ms.get("decoder", 0.0),
            "latency_decision_ms": observation.latency_ms.get("decision", 0.0),
            "latency_latent_ms": observation.latency_ms.get("latent", 0.0),
            "latency_publication_ms": observation.latency_ms.get("publication", 0.0),
            "latency_adaptation_ms": observation.latency_ms.get("adaptation", 0.0),
            "latency_total_compute_ms": observation.latency_ms.get("total_compute", 0.0),
        }

    def _write_alignment_status(self) -> None:
        (self.artifact_dir / "alignment_status.json").write_text(
            json.dumps(self.aligner.snapshot(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_adaptation_log(self) -> None:
        path = self.artifact_dir / "adaptation_log.csv"
        fieldnames = [
            "timestamp",
            "window_id",
            "phase",
            "label",
            "confidence",
            "quality_score",
            "model_version_before",
            "model_version_after",
            "margin",
            "dwell_seconds",
            "accepted",
            "reason",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._adaptation_log)

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
