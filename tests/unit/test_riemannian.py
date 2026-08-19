from __future__ import annotations

import numpy as np

from bci.config import load_config
from bci.domain import FeatureRecord
from bci.models.factory import get_decoder
from bci.models.riemannian import RiemannianMDMDecoder


def covariance_record(label: str, matrix: np.ndarray, trial_id: str) -> FeatureRecord:
    tri_i, tri_j = np.triu_indices(matrix.shape[0])
    return FeatureRecord(
        trial_id=trial_id,
        label=label,
        split="calibration",
        values=matrix[tri_i, tri_j],
        feature_names=[f"cov{i}{j}" for i, j in zip(tri_i, tri_j)],
        frequency_scores={},
        provenance={"end_time": 0.0, "event_index": 0},
        config_hash="test",
        covariance_matrices=matrix[None, :, :],
        covariance_band_names=["broad"],
        representation_type="covariance",
    )


def test_riemannian_mdm_separates_toy_spd_classes():
    config = load_config("configs/kalunga_v0.yaml")
    decoder = RiemannianMDMDecoder(config)
    left = [np.diag([3.0 + 0.1 * i, 1.0, 1.0]) for i in range(3)]
    right = [np.diag([1.0, 3.0 + 0.1 * i, 1.0]) for i in range(3)]
    records = [
        *(covariance_record("LEFT", matrix, f"left-{i}") for i, matrix in enumerate(left)),
        *(covariance_record("RIGHT", matrix, f"right-{i}") for i, matrix in enumerate(right)),
    ]
    decoder.fit(records)
    probs = decoder.predict(covariance_record("LEFT", np.diag([3.2, 1.0, 1.0]), "probe"))
    assert probs["LEFT"] > probs["RIGHT"]
    assert probs["LEFT"] > 0.8
    diagnostics = decoder.diagnostics()
    assert diagnostics.separation["LEFT_vs_RIGHT"] > 0.0


def test_decoder_factory_selects_riemannian_mdm():
    config = load_config("configs/kalunga_v0.yaml")
    config.model.type = "riemannian_mdm"
    assert isinstance(get_decoder(config), RiemannianMDMDecoder)
