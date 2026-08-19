from __future__ import annotations

import numpy as np

from bci.config import load_config
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
