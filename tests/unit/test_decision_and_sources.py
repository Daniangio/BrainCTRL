from __future__ import annotations

from bci.config import load_config
from bci.domain import Prediction
from bci.inference.decision import ExponentialEvidencePolicy
from bci.inference.factory import get_decision_policy
from bci.inference.decision import MarkovEvidencePolicy
from bci.sources.replay import RawReplayEEGSource, RawReplayEventSource
from bci.sources.synthetic import SyntheticEEGSource

import numpy as np


def test_abstention_below_threshold():
    config = load_config("configs/kalunga_v0.yaml")
    policy = ExponentialEvidencePolicy(config)
    decision = policy.update(
        Prediction(
            trial_id="t",
            true_label=None,
            probabilities={"LEFT": 0.5, "RIGHT": 0.3, "NONE": 0.2},
            predicted_label="LEFT",
            confidence=0.5,
            model_version=1,
            timestamp=1.0,
        )
    )
    assert decision.command == "NONE"


def test_markov_evidence_policy_resists_weak_one_window_switch():
    config = load_config("configs/kalunga_v0.yaml")
    config.decision.type = "markov_evidence"
    config.decision.mode = "held_state"
    config.decision.posterior_threshold = 0.55
    config.decision.consecutive_windows = 1
    config.decision.refractory_seconds = 0.0
    policy = MarkovEvidencePolicy(config)
    first = policy.update(
        Prediction(
            trial_id="t1",
            true_label=None,
            probabilities={"LEFT": 0.90, "RIGHT": 0.05, "NONE": 0.05},
            predicted_label="LEFT",
            confidence=0.90,
            model_version=1,
            timestamp=1.0,
        )
    )
    second = policy.update(
        Prediction(
            trial_id="t2",
            true_label=None,
            probabilities={"LEFT": 0.35, "RIGHT": 0.55, "NONE": 0.10},
            predicted_label="RIGHT",
            confidence=0.55,
            model_version=1,
            timestamp=1.25,
        )
    )
    assert first.command == "LEFT"
    assert second.command != "RIGHT"
    assert second.probabilities["LEFT"] > second.probabilities["RIGHT"]


def test_decision_policy_factory_selects_markov():
    config = load_config("configs/kalunga_v0.yaml")
    config.decision.type = "markov_evidence"
    assert isinstance(get_decision_policy(config), MarkovEvidencePolicy)


def test_source_substitution_contract():
    a = SyntheticEEGSource()
    b = SyntheticEEGSource(ch_names=["O1"])
    for source in (a, b):
        meta = source.connect()
        chunk = source.read_latest(1.0)
        assert chunk.data.shape[0] == len(meta.ch_names)
        source.close()


class _FakeRaw:
    def __init__(self):
        self.info = {"sfreq": 10.0}
        self.ch_names = ["Oz", "Stim"]
        self.n_times = 30
        self.annotations = [
            {"onset": 0.1, "duration": 1.0, "description": "13"},
            {"onset": 0.2, "duration": 1.0, "description": "17"},
            {"onset": 0.3, "duration": 1.0, "description": "rest"},
        ]
        self._data = np.vstack([np.arange(self.n_times, dtype=float), np.zeros(self.n_times)])

    def get_channel_types(self):
        return ["eeg", "stim"]

    def get_data(self, picks, start, stop):
        indices = [self.ch_names.index(pick) for pick in picks]
        return self._data[indices, start:stop]


def test_raw_replay_sources_preserve_raw_annotation_indices():
    config = load_config("configs/kalunga_v0.yaml")
    raw = _FakeRaw()
    eeg = RawReplayEEGSource(config, raw)
    meta = eeg.connect()
    chunk = eeg.poll_new()
    assert meta.ch_names == ["Oz"]
    assert chunk is not None
    assert chunk.data.shape == (1, config.source.replay.chunk_size_samples)
    events = RawReplayEventSource(config, raw)
    events.connect()
    events.advance_to(1.0)
    decoded = events.poll()
    assert [(event.native_label, event.command, event.event_index) for event in decoded] == [
        ("13", "LEFT", 0),
        ("17", None, 1),
        ("rest", "NONE", 2),
    ]
