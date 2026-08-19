from __future__ import annotations

from bci.config import BCIConfig
from bci.features.base import FeatureExtractor
from bci.features.cca import FBCCAFeatureExtractor
from bci.features.covariance import CovarianceFeatureExtractor
from bci.features.spectral import SpectralFeatureExtractor


def get_feature_extractor(config: BCIConfig) -> FeatureExtractor:
    registry = {
        "spectral_relative_power": SpectralFeatureExtractor,
        "fbcca": FBCCAFeatureExtractor,
        "covariance": CovarianceFeatureExtractor,
    }
    try:
        return registry[config.features.type](config)
    except KeyError as exc:
        raise ValueError(f"unsupported feature extractor {config.features.type!r}; available: {sorted(registry)}") from exc
