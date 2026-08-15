from __future__ import annotations

from bci.config import load_config
from bci.domain import Prediction
from bci.inference.decision import ExponentialEvidencePolicy
from bci.sources.synthetic import SyntheticEEGSource


def test_abstention_below_threshold():
    config = load_config("configs/kalunga_v0.yaml")
    policy = ExponentialEvidencePolicy(config)
    decision = policy.update(
        Prediction(
            trial_id="t",
            true_label=None,
            probabilities={"LEFT": 0.5, "RIGHT": 0.3, "NONE": 0.2},
            predicted_label="LEFT",
            confidence=0.5,
            model_version=1,
            timestamp=1.0,
        )
    )
    assert decision.command == "NONE"


def test_source_substitution_contract():
    a = SyntheticEEGSource()
    b = SyntheticEEGSource(ch_names=["O1"])
    for source in (a, b):
        meta = source.connect()
        chunk = source.read_latest(1.0)
        assert chunk.data.shape[0] == len(meta.ch_names)
        source.close()
