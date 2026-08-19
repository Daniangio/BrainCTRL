from __future__ import annotations

import numpy as np

from bci.adaptation.riemannian import RiemannianPrototypeAdaptor
from bci.config import load_config
from bci.domain import CalibrationPhase, Decision, FeatureRecord, OnlineObservation, Prediction, SignalQuality
from bci.models.riemannian import RiemannianMDMDecoder


def covariance_feature(label: str, matrix: np.ndarray, trial_id: str) -> FeatureRecord:
    tri_i, tri_j = np.triu_indices(matrix.shape[0])
    return FeatureRecord(
        trial_id=trial_id,
        label=label,
        split="calibration",
        values=matrix[tri_i, tri_j],
        feature_names=[f"cov{i}{j}" for i, j in zip(tri_i, tri_j)],
        frequency_scores={},
        provenance={"start_time": 0.0, "end_time": 1.0, "event_index": 0},
        config_hash="test",
        covariance_matrices=matrix[None, :, :],
        covariance_band_names=["broad"],
        representation_type="covariance",
    )


def online_observation(feature: FeatureRecord, decision: Decision, quality: SignalQuality) -> OnlineObservation:
    prediction = Prediction(
        trial_id=feature.trial_id,
        true_label=None,
        probabilities=dict(decision.probabilities),
        predicted_label=decision.command,
        confidence=decision.confidence,
        model_version=decision.model_version,
        timestamp=decision.timestamp,
    )
    return OnlineObservation(
        window_id=feature.trial_id,
        window_start=0.0,
        window_end=decision.timestamp,
        phase=CalibrationPhase.INFERENCE,
        feature=feature,
        prediction=prediction,
        evidence_prediction=prediction,
        decision=decision,
        quality=quality,
        current_ground_truth_if_known=None,
        model_version=decision.model_version,
        emitted=True,
    )


def fitted_decoder() -> RiemannianMDMDecoder:
    config = load_config("configs/kalunga_v0.yaml")
    decoder = RiemannianMDMDecoder(config)
    decoder.fit(
        [
            covariance_feature("LEFT", np.diag([3.0, 1.0]), "left-1"),
            covariance_feature("LEFT", np.diag([3.2, 1.0]), "left-2"),
            covariance_feature("RIGHT", np.diag([1.0, 3.0]), "right-1"),
            covariance_feature("RIGHT", np.diag([1.0, 3.2]), "right-2"),
        ]
    )
    return decoder


def test_riemannian_prototype_adaptor_updates_guarded_class_center():
    config = load_config("configs/kalunga_v0.yaml")
    config.adaptation.enabled = True
    config.adaptation.min_temporal_posterior = 0.8
    config.adaptation.min_margin = 0.5
    config.adaptation.min_dwell_seconds = 0.0
    config.adaptation.max_updates_per_second = 10.0
    config.adaptation.eta = 0.05
    decoder = fitted_decoder()
    before = decoder.diagnostics().class_centers["LEFT"].copy()
    adaptor = RiemannianPrototypeAdaptor(config)
    decision = Decision(
        timestamp=2.0,
        command="LEFT",
        probabilities={"LEFT": 0.95, "RIGHT": 0.03, "NONE": 0.02},
        confidence=0.95,
        model_version=decoder.model_version,
        reason="markov_emitted",
    )
    quality = SignalQuality(score=1.0, flags=[], per_channel={}, metrics={}, history_ready=True)
    row = adaptor.update(online_observation(covariance_feature("LEFT", np.diag([4.0, 1.0]), "online"), decision, quality), decoder)
    after = decoder.diagnostics().class_centers["LEFT"]
    assert row["accepted"] is True
    assert row["reason"] == "updated"
    assert not np.allclose(before, after)


def test_riemannian_prototype_adaptor_rejects_low_quality_window():
    config = load_config("configs/kalunga_v0.yaml")
    config.adaptation.enabled = True
    decoder = fitted_decoder()
    adaptor = RiemannianPrototypeAdaptor(config)
    decision = Decision(
        timestamp=2.0,
        command="LEFT",
        probabilities={"LEFT": 0.99, "RIGHT": 0.005, "NONE": 0.005},
        confidence=0.99,
        model_version=decoder.model_version,
        reason="markov_emitted",
    )
    quality = SignalQuality(score=0.2, flags=["hard_reject"], per_channel={}, metrics={}, history_ready=True)
    row = adaptor.update(online_observation(covariance_feature("LEFT", np.diag([4.0, 1.0]), "online"), decision, quality), decoder)
    assert row["accepted"] is False
    assert row["reason"] == "low_quality"
