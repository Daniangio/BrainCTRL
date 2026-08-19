from __future__ import annotations

from bci.config import BCIConfig
from bci.models.base import Decoder
from bci.models.cca import CCADecoder
from bci.models.gaussian_latent import GaussianLatentDecoder
from bci.models.riemannian import RiemannianMDMDecoder
from bci.models.spectral_score import SpectralScoreDecoder


def get_decoder(config: BCIConfig) -> Decoder:
    name = config.model.type
    if name == "bayesian_latent":
        name = "gaussian_latent"
    registry = {
        "gaussian_latent": GaussianLatentDecoder,
        "spectral_score": SpectralScoreDecoder,
        "cca": CCADecoder,
        "riemannian_mdm": RiemannianMDMDecoder,
    }
    try:
        return registry[name](config)
    except KeyError as exc:
        raise ValueError(f"unsupported decoder {config.model.type!r}; available: {sorted(registry)}") from exc
