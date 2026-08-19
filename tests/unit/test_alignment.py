from __future__ import annotations

import numpy as np

from bci.config import load_config
from bci.domain import FeatureRecord, SignalQuality
from bci.features.alignment import EuclideanAlignment


def covariance_feature(matrix: np.ndarray, trial_id: str = "cov", start: float = 0.0, end: float = 2.0) -> FeatureRecord:
    tri_i, tri_j = np.triu_indices(matrix.shape[0])
    return FeatureRecord(
        trial_id=trial_id,
        label="LEFT",
        split="calibration",
        values=matrix[tri_i, tri_j],
        feature_names=[f"cov{i}{j}" for i, j in zip(tri_i, tri_j)],
        frequency_scores={},
        provenance={"start_time": start, "end_time": end, "event_index": 0},
        config_hash="test",
        covariance_matrices=matrix[None, :, :],
        covariance_band_names=["broad"],
        representation_type="covariance",
    )


def test_euclidean_alignment_maps_reference_covariance_near_identity():
    config = load_config("configs/kalunga_v0.yaml")
    config.alignment.enabled = True
    config.alignment.type = "euclidean"
    aligner = EuclideanAlignment(config)
    aligned = aligner.update_transform(covariance_feature(np.diag([4.0, 1.0])))
    assert aligned.covariance_matrices is not None
    assert aligned.alignment_version == 1
    assert np.allclose(aligned.covariance_matrices[0], np.eye(2), atol=1.0e-5)


def test_euclidean_alignment_skips_low_quality_updates():
    config = load_config("configs/kalunga_v0.yaml")
    config.alignment.enabled = True
    config.alignment.type = "euclidean"
    aligner = EuclideanAlignment(config)
    quality = SignalQuality(score=0.1, flags=["hard_reject"], per_channel={}, metrics={}, history_ready=True)
    feature = covariance_feature(np.diag([4.0, 1.0]))
    unchanged = aligner.update_transform(feature, quality)
    assert aligner.version == 0
    assert unchanged is feature


def test_euclidean_alignment_freezes_after_warmup():
    config = load_config("configs/kalunga_v0.yaml")
    config.alignment.enabled = True
    config.alignment.type = "euclidean"
    config.alignment.warmup_seconds = 1.0
    aligner = EuclideanAlignment(config)
    first = aligner.update_transform(covariance_feature(np.diag([4.0, 1.0]), "first", 0.0, 1.0))
    version = aligner.version
    second = aligner.update_transform(covariance_feature(np.diag([1.0, 4.0]), "second", 1.0, 2.0))
    assert aligner.frozen
    assert aligner.version == version
    assert first.alignment_version == second.alignment_version
