from __future__ import annotations

import numpy as np

from bci.buffering.ring import TimestampedRingBuffer
from bci.config import load_config
from bci.domain import BCIEvent, EEGChunk
from bci.experiment.bus import EventBus
from bci.experiment.events import CalibrationStatus, DecisionEmitted, ModelUpdated, PredictionProduced
from bci.experiment.factory import build_realtime_experiment
from bci.experiment.trial_builder import RealtimeTrialBuilder
from bci.features.spectral import SpectralFeatureExtractor
from bci.preprocessing.standard import StandardPreprocessor


def test_synthetic_realtime_engine_headless(tmp_path):
    config = load_config("configs/kalunga_v0.yaml")
    config.experiment.mode = "synthetic"
    config.experiment.gui = False
    config.gui.enabled = False
    config.output.console = False
    config.experiment.max_idle_seconds = 2.0
    managed = build_realtime_experiment(config, artifact_dir=tmp_path)
    result = managed.run()
    assert result.n_trials > 0
    assert (tmp_path / "features.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "protocol_manifest.csv").exists()


def test_classifier_smoke_is_interpretable_and_trains_headlessly(tmp_path):
    config = load_config("configs/kalunga_v0.yaml")
    config.experiment.mode = "classifier_smoke"
    config.experiment.gui = False
    config.gui.enabled = False
    config.output.console = False
    config.experiment.max_idle_seconds = 2.0
    bus = EventBus()
    statuses: list[CalibrationStatus] = []
    updates: list[ModelUpdated] = []
    predictions: list[PredictionProduced] = []
    bus.subscribe(CalibrationStatus, statuses.append)
    bus.subscribe(ModelUpdated, updates.append)
    bus.subscribe(PredictionProduced, predictions.append)
    result = build_realtime_experiment(config, bus=bus, artifact_dir=tmp_path).run()
    assert result.model_version >= 1
    assert result.metrics["smoke"]["mode"] == "classifier_smoke"
    assert "Gaussian latent" in result.metrics["smoke"]["purpose"]
    assert updates
    assert any("waiting for" in status.reason for status in statuses)
    assert any("TRAINED" in status.reason for status in statuses)
    assert len(predictions) == result.metrics["validation"]["n"] + result.metrics["test"]["n"]
    assert result.metrics["test"]["balanced_accuracy"] >= 0.9


def test_controller_smoke_exercises_evidence_accumulator(tmp_path):
    config = load_config("configs/kalunga_v0.yaml")
    config.experiment.mode = "controller_smoke"
    config.experiment.gui = False
    config.gui.enabled = False
    config.output.console = False
    config.experiment.max_idle_seconds = 2.0
    config.decision.consecutive_windows = 2
    bus = EventBus()
    decisions: list[DecisionEmitted] = []
    bus.subscribe(DecisionEmitted, decisions.append)
    result = build_realtime_experiment(config, bus=bus, artifact_dir=tmp_path).run()
    emitted = [event.decision.command for event in decisions if event.decision.command != "NONE"]
    assert result.model_version >= 1
    assert result.metrics["smoke"]["mode"] == "controller_smoke"
    assert result.metrics["smoke"]["decision_policy"]["consecutive_windows"] == 2
    assert result.metrics["smoke"]["n_calibration_features"] == 60
    assert "LEFT" in emitted
    assert "RIGHT" in emitted
    assert any(event.decision.reason.startswith("waiting_consecutive") for event in decisions)
    assert "reason" in (tmp_path / "decisions.csv").read_text(encoding="utf-8").splitlines()[0]


def test_replay_features_match_offline_features_synthetic():
    config = load_config("configs/kalunga_v0.yaml")
    config.preprocessing.bandpass_hz = None
    config.preprocessing.notch_hz = None
    config.trials.onset_offset_seconds = 0.25
    config.trials.window_seconds = 1.5
    sfreq = 128.0
    times = np.arange(int(3.0 * sfreq)) / sfreq
    data = np.sin(2 * np.pi * 13.0 * times)[None, :]
    event = BCIEvent(
        timestamp=0.5,
        duration=2.0,
        native_label="13",
        command="LEFT",
        event_index=0,
        dataset="Synthetic",
        subject=1,
        session="0",
        run="0",
    )
    start = event.timestamp + config.trials.onset_offset_seconds
    end = start + config.trials.window_seconds
    start_idx = int(round(start * sfreq))
    stop_idx = int(round(end * sfreq))
    from bci.domain import TrialRecord

    offline = TrialRecord(
        trial_id="Synthetic-s1-0-0-e0",
        dataset="Synthetic",
        subject=1,
        session="0",
        run="0",
        event_index=0,
        native_label="13",
        command="LEFT",
        start_time=start,
        end_time=end,
        sfreq=sfreq,
        ch_names=["Oz"],
        data=data[:, start_idx:stop_idx],
        split="calibration",
    )
    ring = TimestampedRingBuffer(max_seconds=4.0, sfreq=sfreq, ch_names=["Oz"])
    ring.append(EEGChunk(data=data, sfreq=sfreq, ch_names=["Oz"], t_start=0.0, times=times))
    builder = RealtimeTrialBuilder(config, {0: "calibration"})
    builder.add_event(event)
    realtime = builder.resolve(ring)[0]
    pre = StandardPreprocessor(config)
    extractor = SpectralFeatureExtractor(config)
    offline_features = extractor.transform(pre.transform(offline)).values
    realtime_features = extractor.transform(pre.transform(realtime)).values
    assert np.allclose(realtime.data, offline.data, atol=1e-12)
    assert np.allclose(realtime_features, offline_features, atol=1e-10)
