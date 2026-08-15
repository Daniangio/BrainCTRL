from __future__ import annotations

from pathlib import Path

import numpy as np

from bci.buffering.ring import TimestampedRingBuffer
from bci.config import load_config
from bci.domain import BCIEvent, EEGChunk
from bci.experiment.bus import EventBus
from bci.experiment.events import TrialStarted
from bci.experiment.trial_builder import RealtimeTrialBuilder
from bci.models.bayesian_latent import BayesianLatentDecoder
from bci.sources.events import decode_one_hot_annotations
from tests.unit.test_decoder import records


def test_annotation_decoding_one_hot():
    config = load_config("configs/kalunga_v0.yaml")
    data = np.asarray([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]])
    times = np.asarray([10.0, 12.0, 14.0])
    events = decode_one_hot_annotations(config, data, times, ["13", "17", "rest"])
    assert [(e.native_label, e.command, e.event_index) for e in events] == [
        ("13", "LEFT", 0),
        ("17", None, 1),
        ("rest", "NONE", 2),
    ]


def test_timestamped_ring_buffer_slicing():
    data = np.arange(20, dtype=float)[None, :]
    times = np.arange(20, dtype=float) / 10.0
    chunk = EEGChunk(data=data, sfreq=10.0, ch_names=["Oz"], t_start=0.0, times=times)
    ring = TimestampedRingBuffer(max_seconds=3.0, sfreq=10.0, ch_names=["Oz"])
    ring.append(chunk)
    sliced = ring.slice(0.5, 1.0, expected_samples=5)
    assert sliced.data.tolist() == [[5.0, 6.0, 7.0, 8.0, 9.0]]


def test_pending_trial_resolution():
    config = load_config("configs/kalunga_v0.yaml")
    config.trials.onset_offset_seconds = 0.0
    config.trials.window_seconds = 0.5
    ring = TimestampedRingBuffer(max_seconds=2.0, sfreq=10.0, ch_names=["Oz"])
    builder = RealtimeTrialBuilder(config, {0: "calibration"})
    event = BCIEvent(0.5, 0.5, "13", "LEFT", event_index=0, dataset="Synthetic", subject=1, session="0", run="0")
    builder.add_event(event)
    ring.append(EEGChunk(np.arange(20, dtype=float)[None, :], 10.0, ["Oz"], 0.0, np.arange(20) / 10.0))
    trials = builder.resolve(ring)
    assert len(trials) == 1
    assert trials[0].split == "calibration"
    assert trials[0].data.shape == (1, 5)


def test_pending_trial_respects_allowed_splits():
    config = load_config("configs/kalunga_v0.yaml")
    config.trials.onset_offset_seconds = 0.0
    config.trials.window_seconds = 0.5
    builder = RealtimeTrialBuilder(config, {0: "calibration", 1: "validation"})
    calibration = BCIEvent(0.5, 0.5, "13", "LEFT", event_index=0, dataset="Synthetic", subject=1, session="0", run="0")
    validation = BCIEvent(1.0, 0.5, "21", "RIGHT", event_index=1, dataset="Synthetic", subject=1, session="0", run="0")
    assert builder.add_event(calibration, allowed_splits={"validation"}) is None
    assert builder.add_event(validation, allowed_splits={"validation"}) is not None


def test_event_bus_delivery():
    bus = EventBus()
    seen = []
    event = BCIEvent(0.0, 1.0, "13", "LEFT")
    bus.subscribe(TrialStarted, seen.append)
    bus.publish(TrialStarted(event))
    assert seen == [TrialStarted(event)]


def test_decoder_diagnostics_api():
    decoder = BayesianLatentDecoder(load_config("configs/kalunga_v0.yaml"))
    decoder.fit(records())
    diagnostics = decoder.diagnostics(records())
    assert diagnostics.model_version == 1
    assert diagnostics.latent_points is not None
    assert diagnostics.class_centers


def test_cli_config_overrides():
    from argparse import Namespace

    from bci.cli import apply_overrides

    config = load_config("configs/kalunga_v0.yaml")
    out = apply_overrides(
        config,
        Namespace(
            subject=2,
            model="bayesian_latent",
            max_trials=4,
            gui=False,
            no_gui=True,
            synthetic=True,
            smoke_mode=None,
            synthetic_difficulty="perfect",
        ),
    )
    assert out.dataset.subjects == [2]
    assert out.experiment.max_trials == 4
    assert out.experiment.mode == "classifier_smoke"
    assert out.experiment.synthetic_difficulty == "perfect"
    assert not out.gui.enabled
