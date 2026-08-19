from __future__ import annotations

import numpy as np

from bci.config import load_config
from bci.domain import TrialRecord
from bci.features.cca import FBCCAFeatureExtractor
from bci.models.cca import CCADecoder


def make_multichannel_sine(freq: float, command: str = "LEFT", sfreq: float = 256.0, seconds: float = 2.0) -> TrialRecord:
    times = np.arange(int(sfreq * seconds)) / sfreq
    data = np.vstack(
        [
            np.sin(2 * np.pi * freq * times),
            0.7 * np.sin(2 * np.pi * freq * times + 0.5),
        ]
    )
    return TrialRecord(
        trial_id=f"synthetic-{freq:g}",
        dataset="synthetic",
        subject=1,
        session="0",
        run="0",
        event_index=0,
        native_label=str(freq),
        command=command,
        start_time=0.0,
        end_time=seconds,
        sfreq=sfreq,
        ch_names=["Oz", "O1"],
        data=data,
        split="test",
    )


def test_fbcca_scores_target_frequency_highest():
    config = load_config("configs/kalunga_v0.yaml")
    config.features.type = "fbcca"
    feature = FBCCAFeatureExtractor(config).transform(make_multichannel_sine(13.0))
    assert feature.frequency_scores["LEFT"] > feature.frequency_scores["RIGHT"]
    assert feature.spectral_power is not None
    assert feature.spectral_channel_names == ["Oz", "O1"]


def test_cca_decoder_is_calibration_free_and_uses_fbcca_scores():
    config = load_config("configs/kalunga_v0.yaml")
    config.features.type = "fbcca"
    config.model.type = "cca"
    feature = FBCCAFeatureExtractor(config).transform(make_multichannel_sine(21.0, command="RIGHT"))
    decoder = CCADecoder(config)
    probs = decoder.predict(feature)
    assert decoder.model_version == 1
    assert probs["RIGHT"] > probs["LEFT"]
    assert probs["RIGHT"] > probs["NONE"]


def test_cca_decoder_abstains_when_fbcca_activation_is_weak():
    config = load_config("configs/kalunga_v0.yaml")
    config.features.type = "fbcca"
    config.model.type = "cca"
    feature = FBCCAFeatureExtractor(config).transform(make_multichannel_sine(7.0, command="NONE"))
    probs = CCADecoder(config).predict(feature)
    assert probs["NONE"] > probs["LEFT"]
    assert probs["NONE"] > probs["RIGHT"]
