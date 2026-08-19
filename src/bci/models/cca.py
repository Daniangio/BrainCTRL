from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import softmax

from bci.config import BCIConfig
from bci.domain import FeatureRecord
from bci.models.base import Decoder


class CCADecoder(Decoder):
    def __init__(self, config: BCIConfig):
        self.config = config
        self.model_version = 1
        self._classes = list(config.protocol.classes)
        self.activation_threshold_ = config.model.cca_activation_threshold

    @property
    def classes_(self) -> Sequence[str]:
        return self._classes

    def fit(self, records: Sequence[FeatureRecord]) -> None:
        if records:
            self._classes = sorted(set(r.label for r in records) | set(self.config.protocol.classes))
            rest = [r for r in records if r.label == self.config.commands.reject_command]
            if rest:
                rest_max = float(np.median([max(r.frequency_scores.values(), default=0.0) for r in rest]))
                self.activation_threshold_ = max(self.config.model.cca_activation_threshold, 2.0 * rest_max)
        self.model_version = max(1, self.model_version + 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CCADecoder predicts from FBCCA FeatureRecord scores")

    def predict(self, features: FeatureRecord) -> dict[str, float]:
        scale = self.config.model.cca_logit_scale
        raw = []
        for cls in self._classes:
            if cls == self.config.commands.reject_command:
                raw.append(0.0)
            else:
                raw.append(scale * (features.frequency_scores.get(cls, 0.0) - self.activation_threshold_))
        probs = softmax(np.asarray(raw, dtype=float))
        return dict(zip(self._classes, map(float, probs)))

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "CCADecoder":
        with path.open("rb") as f:
            return pickle.load(f)
