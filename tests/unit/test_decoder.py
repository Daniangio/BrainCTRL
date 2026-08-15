from __future__ import annotations

from pathlib import Path

import numpy as np

from bci.config import load_config
from bci.domain import FeatureRecord
from bci.models.bayesian_latent import BayesianLatentDecoder


def rec(i: int, label: str, x: list[float]) -> FeatureRecord:
    return FeatureRecord(
        trial_id=f"{label}-{i}",
        label=label,
        split="calibration",
        values=np.asarray(x, dtype=float),
        feature_names=["a", "b"],
        frequency_scores={},
        provenance={"event_index": i},
        config_hash="h",
    )


def records():
    return [
        rec(0, "LEFT", [2.0, 0.1]),
        rec(1, "LEFT", [2.2, 0.2]),
        rec(2, "RIGHT", [-2.0, 0.0]),
        rec(3, "RIGHT", [-2.1, 0.1]),
        rec(4, "NONE", [0.0, -1.0]),
        rec(5, "NONE", [0.1, -1.1]),
    ]


def test_bayesian_decoder_probabilities_sum_to_one():
    decoder = BayesianLatentDecoder(load_config("configs/kalunga_v0.yaml"))
    decoder.fit(records())
    probs = decoder.predict_proba(np.asarray([[2.0, 0.2]]))[0]
    assert np.isclose(probs.sum(), 1.0)


def test_fisher_objective_is_scale_invariant():
    X = np.asarray([[0.0], [1.0], [3.0], [4.0]])
    y = np.asarray(["A", "A", "B", "B"])
    w = np.asarray([1.0])
    assert np.isclose(
        BayesianLatentDecoder.fisher_objective(X, y, w),
        BayesianLatentDecoder.fisher_objective(X, y, 10.0 * w),
    )


def test_batch_updates_increment_model_version_and_serialization(tmp_path: Path):
    decoder = BayesianLatentDecoder(load_config("configs/kalunga_v0.yaml"))
    decoder.fit(records())
    before = decoder.predict_proba(np.asarray([[2.0, 0.2]]))
    assert decoder.model_version == 1
    decoder.update(records())
    assert decoder.model_version == 2
    path = tmp_path / "model.pkl"
    decoder.save(path)
    loaded = BayesianLatentDecoder.load(path)
    after = loaded.predict_proba(np.asarray([[2.0, 0.2]]))
    assert np.allclose(after, decoder.predict_proba(np.asarray([[2.0, 0.2]])))
    assert before.shape == after.shape
