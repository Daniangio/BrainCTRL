from __future__ import annotations

from bci.config import BCIConfig
from bci.models.spectral_score import SpectralScoreDecoder


class CCADecoder(SpectralScoreDecoder):
    """V0-compatible CCA baseline placeholder.

    The class preserves the decoder API and is intentionally isolated so a
    full raw-window CCA implementation can replace it without touching callers.
    """

    def __init__(self, config: BCIConfig):
        super().__init__(config)
