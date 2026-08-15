from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import softmax

from bci.config import BCIConfig
from bci.domain import FeatureRecord
from bci.models.base import Decoder


class SpectralScoreDecoder(Decoder):
    def __init__(self, config: BCIConfig):
        self.config = config
        self.model_version = 0
        self._classes = ["LEFT", "RIGHT", "NONE"]
        self.none_bias_ = 0.0

    @property
    def classes_(self) -> Sequence[str]:
        return self._classes

    def fit(self, records: Sequence[FeatureRecord]) -> None:
        self._classes = sorted(set(r.label for r in records) | {"NONE"})
        active = [r for r in records if r.label != "NONE"]
        rest = [r for r in records if r.label == "NONE"]
        if active and rest:
            self.none_bias_ = float(np.mean([max(r.frequency_scores.values()) for r in rest]))
        self.model_version += 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("SpectralScoreDecoder predicts from FeatureRecord metadata")

    def predict(self, features: FeatureRecord) -> dict[str, float]:
        left = features.frequency_scores.get("LEFT", 0.0)
        right = features.frequency_scores.get("RIGHT", 0.0)
        none = self.none_bias_ - max(left, right)
        raw = np.asarray([left, right, none], dtype=float)
        probs = softmax(raw)
        return dict(zip(["LEFT", "RIGHT", "NONE"], map(float, probs)))

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SpectralScoreDecoder":
        with path.open("rb") as f:
            return pickle.load(f)
