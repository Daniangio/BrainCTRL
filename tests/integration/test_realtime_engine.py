from __future__ import annotations

import numpy as np

from bci.buffering.ring import TimestampedRingBuffer
from bci.config import load_config
from bci.domain import BCIEvent, EEGChunk
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
