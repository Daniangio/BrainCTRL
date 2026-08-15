from __future__ import annotations

from bci.models.bayesian_latent import BayesianLatentDecoder


class GaussianLatentDecoder(BayesianLatentDecoder):
    """Regularized latent Gaussian decoder.

    The historical ``BayesianLatentDecoder`` name remains as a compatibility
    alias. This implementation outputs probabilistic Gaussian class posteriors
    in latent space; it does not yet integrate uncertainty over model
    parameters.
    """
