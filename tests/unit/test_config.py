from __future__ import annotations

from bci.config import load_config


def test_config_validation_and_mapping():
    config = load_config("configs/kalunga_v0.yaml")
    assert config.dataset.name == "Kalunga2016"
    assert config.stimulus_frequencies == {"LEFT": 13.0, "RIGHT": 21.0}


def test_v2_configs_load_expected_backends():
    fbcca = load_config("configs/kalunga_v2_fbcca.yaml")
    covariance = load_config("configs/kalunga_v2_covariance.yaml")
    assert fbcca.features.type == "fbcca"
    assert fbcca.model.type == "cca"
    assert fbcca.decision.type == "markov_evidence"
    assert covariance.features.type == "covariance"
    assert covariance.model.type == "riemannian_mdm"
    assert covariance.alignment.enabled
