from __future__ import annotations

from bci.config import load_config


def test_config_validation_and_mapping():
    config = load_config("configs/kalunga_v0.yaml")
    assert config.dataset.name == "Kalunga2016"
    assert config.stimulus_frequencies == {"LEFT": 13.0, "RIGHT": 21.0}
