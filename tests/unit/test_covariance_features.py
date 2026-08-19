from __future__ import annotations

import numpy as np

from bci.config import load_config
from bci.domain import TrialRecord
from bci.features.covariance import CovarianceFeatureExtractor
from bci.features.factory import get_feature_extractor


def make_covariance_trial(sfreq: float = 128.0, seconds: float = 2.0) -> TrialRecord:
    times = np.arange(int(sfreq * seconds)) / sfreq
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10.0 * times),
            0.5 * np.sin(2 * np.pi * 12.0 * times + 0.2),
            0.2 * np.sin(2 * np.pi * 18.0 * times + 0.7),
        ]
    )
    return TrialRecord(
        trial_id="covariance-synthetic",
        dataset="synthetic",
        subject=1,
        session="0",
        run="0",
        event_index=0,
        native_label="13",
        command="LEFT",
        start_time=0.0,
        end_time=seconds,
        sfreq=sfreq,
        ch_names=["Oz", "O1", "O2"],
        data=data,
        split="calibration",
    )


def test_covariance_extractor_outputs_spd_matrices_and_vector_view():
    config = load_config("configs/kalunga_v0.yaml")
    config.features.type = "covariance"
    record = CovarianceFeatureExtractor(config).transform(make_covariance_trial())
    assert record.representation_type == "covariance"
    assert record.covariance_matrices is not None
    assert record.covariance_matrices.shape == (1, 3, 3)
    assert record.values.shape == (6,)
    assert record.covariance_band_names == ["6-50Hz"]
    eigvals = np.linalg.eigvalsh(record.covariance_matrices[0])
    assert np.all(eigvals > 0.0)


def test_covariance_trace_normalization_preserves_channel_dimension_trace():
    config = load_config("configs/kalunga_v0.yaml")
    config.features.type = "covariance"
    config.features.covariance.normalize = "trace"
    record = CovarianceFeatureExtractor(config).transform(make_covariance_trial())
    assert record.covariance_matrices is not None
    assert np.isclose(np.trace(record.covariance_matrices[0]), 3.0)


def test_feature_factory_selects_covariance_extractor():
    config = load_config("configs/kalunga_v0.yaml")
    config.features.type = "covariance"
    assert isinstance(get_feature_extractor(config), CovarianceFeatureExtractor)
