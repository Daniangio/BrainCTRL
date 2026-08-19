from __future__ import annotations

import numpy as np

from bci.config import load_config
from bci.domain import Prediction, SignalQuality
from bci.inference.decision import ExponentialEvidencePolicy
from bci.inference.quality import quality_adjust_prediction
from bci.preprocessing.quality import SignalQualityEstimator


def test_signal_quality_flags_flat_channels():
    config = load_config("configs/kalunga_v0.yaml")
    estimator = SignalQualityEstimator(config)
    sfreq = 128.0
    times = np.arange(int(2.0 * sfreq)) / sfreq
    clean = np.vstack(
        [
            np.sin(2 * np.pi * 13.0 * times),
            0.8 * np.sin(2 * np.pi * 21.0 * times),
        ]
    )
    clean_quality = estimator.estimate(clean, sfreq, ["Oz", "O1"])
    dropout = clean.copy()
    dropout[1, :] = 0.0
    dropout_quality = estimator.estimate(dropout, sfreq, ["Oz", "O1"])
    assert clean_quality.score > dropout_quality.score
    assert "flat_channel" in dropout_quality.flags
    assert dropout_quality.per_channel["O1"] == 0.0


def test_signal_quality_reports_line_noise_ratio():
    config = load_config("configs/kalunga_v0.yaml")
    estimator = SignalQualityEstimator(config)
    sfreq = 256.0
    times = np.arange(int(2.0 * sfreq)) / sfreq
    line_noise = np.sin(2 * np.pi * 50.0 * times)[None, :]
    quality = estimator.estimate(line_noise, sfreq, ["Oz"])
    assert quality.metrics["line_noise_ratio"] > 0.9
    assert "line_noise" in quality.flags


def test_quality_adjustment_hard_rejects_to_uniform_evidence():
    prediction = Prediction(
        trial_id="online-1",
        true_label=None,
        probabilities={"LEFT": 0.98, "RIGHT": 0.01, "NONE": 0.01},
        predicted_label="LEFT",
        confidence=0.98,
        model_version=1,
        timestamp=1.0,
    )
    quality = SignalQuality(
        score=0.1,
        flags=["hard_reject"],
        per_channel={"Oz": 0.0},
        metrics={},
        history_ready=True,
    )
    adjusted, action = quality_adjust_prediction(prediction, quality, hard_reject_threshold=0.25)
    assert action == "hard_reject_uniform"
    assert adjusted.predicted_label in {"LEFT", "RIGHT", "NONE"}
    assert all(abs(prob - 1.0 / 3.0) < 1e-12 for prob in adjusted.probabilities.values())
    assert prediction.probabilities["LEFT"] == 0.98


def test_quality_rejected_artifact_does_not_emit_false_switch():
    config = load_config("configs/kalunga_v0.yaml")
    config.decision.alpha = 1.0
    config.decision.posterior_threshold = 0.6
    config.decision.consecutive_windows = 1
    policy = ExponentialEvidencePolicy(config)
    artifact_prediction = Prediction(
        trial_id="online-2",
        true_label=None,
        probabilities={"LEFT": 0.01, "RIGHT": 0.98, "NONE": 0.01},
        predicted_label="RIGHT",
        confidence=0.98,
        model_version=1,
        timestamp=2.0,
    )
    quality = SignalQuality(
        score=0.0,
        flags=["hard_reject"],
        per_channel={"Oz": 0.0},
        metrics={},
        history_ready=True,
    )
    evidence, _ = quality_adjust_prediction(artifact_prediction, quality, config.quality.hard_reject_threshold)
    decision = policy.update(evidence)
    assert decision.command == "NONE"
    assert decision.reason == "below_threshold"
