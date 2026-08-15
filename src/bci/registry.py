from __future__ import annotations

from collections.abc import Callable

from bci.config import BCIConfig
from bci.sources.base import DatasetAdapter
from bci.sources.moabb import Kalunga2016Adapter, Lee2019SSVEPAdapter, Nakanishi2015Adapter


_REGISTRY: dict[str, Callable[[BCIConfig], DatasetAdapter]] = {
    "Kalunga2016": Kalunga2016Adapter,
    "Lee2019_SSVEP": Lee2019SSVEPAdapter,
    "Nakanishi2015": Nakanishi2015Adapter,
}


def dataset_names() -> list[str]:
    return sorted(_REGISTRY)


def get_dataset_adapter(config: BCIConfig) -> DatasetAdapter:
    try:
        return _REGISTRY[config.dataset.name](config)
    except KeyError as exc:
        raise ValueError(f"unknown dataset {config.dataset.name!r}; available: {dataset_names()}") from exc
