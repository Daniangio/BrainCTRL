from __future__ import annotations

import numpy as np

from bci.config import load_config
from bci.domain import TrialRecord
from bci.features.spectral import SpectralFeatureExtractor


def make_sine(freq: float, sfreq: float = 256.0, seconds: float = 1.5) -> TrialRecord:
    t = np.arange(int(sfreq * seconds)) / sfreq
    data = np.sin(2 * np.pi * freq * t)[None, :]
    return TrialRecord(
        trial_id="synthetic",
        dataset="synthetic",
        subject=1,
        session="0",
        run="0",
        event_index=0,
        native_label=str(freq),
        command="LEFT",
        start_time=0.0,
        end_time=seconds,
        sfreq=sfreq,
        ch_names=["Oz"],
        data=data,
        split="calibration",
    )


def test_spectral_extractor_identifies_sinusoid():
    config = load_config("configs/kalunga_v0.yaml")
    record = SpectralFeatureExtractor(config).transform(make_sine(13.0))
    assert record.frequency_scores["LEFT"] > record.frequency_scores["RIGHT"]


def test_harmonics_above_nyquist_are_excluded():
    config = load_config("configs/kalunga_v0.yaml")
    trial = make_sine(21.0, sfreq=80.0)
    record = SpectralFeatureExtractor(config).transform(trial)
    assert all("21Hz:h2" not in name and "21Hz:h3" not in name for name in record.feature_names)
