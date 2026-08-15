from __future__ import annotations

from bci.config import load_config
from bci.registry import dataset_names, get_dataset_adapter


def test_dataset_registry_lookup():
    config = load_config("configs/kalunga_v0.yaml")
    assert "Kalunga2016" in dataset_names()
    assert get_dataset_adapter(config).__class__.__name__ == "Kalunga2016Adapter"
