from __future__ import annotations

import numpy as np

from bci.config import load_config
from bci.domain import FeatureRecord
from bci.models.factory import get_decoder
from bci.models.riemannian import RiemannianMDMDecoder


def covariance_record(label: str, matrix: np.ndarray, trial_id: str) -> FeatureRecord:
    matrices = matrix[None, :, :] if matrix.ndim == 2 else matrix
    tri_i, tri_j = np.triu_indices(matrices.shape[1])
    return FeatureRecord(
        trial_id=trial_id,
        label=label,
        split="calibration",
        values=np.concatenate([band[tri_i, tri_j] for band in matrices]),
        feature_names=[f"cov{i}{j}" for i, j in zip(tri_i, tri_j)],
        frequency_scores={},
        provenance={"end_time": 0.0, "event_index": 0},
        config_hash="test",
        covariance_matrices=matrices,
        covariance_band_names=[f"band{i + 1}" for i in range(matrices.shape[0])],
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


def test_riemannian_mdm_fuses_multiple_covariance_bands():
    config = load_config("configs/kalunga_v0.yaml")
    decoder = RiemannianMDMDecoder(config)
    shared = np.eye(2)
    left_band = np.diag([4.0, 1.0])
    right_band = np.diag([1.0, 4.0])
    records = [
        covariance_record("LEFT", np.asarray([shared, left_band]), "left-1"),
        covariance_record("LEFT", np.asarray([shared, np.diag([4.2, 1.0])]), "left-2"),
        covariance_record("RIGHT", np.asarray([shared, right_band]), "right-1"),
        covariance_record("RIGHT", np.asarray([shared, np.diag([1.0, 4.2])]), "right-2"),
    ]
    decoder.fit(records)
    probs = decoder.predict(covariance_record("RIGHT", np.asarray([shared, np.diag([1.0, 4.1])]), "probe"))
    assert probs["RIGHT"] > probs["LEFT"]
    assert "band2:LEFT_vs_RIGHT" in decoder.diagnostics().separation
