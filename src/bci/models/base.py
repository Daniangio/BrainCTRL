from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from bci.domain import FeatureRecord


class Decoder(ABC):
    model_version: int = 0

    @abstractmethod
    def fit(self, records: Sequence[FeatureRecord]) -> None: ...

    def update(self, records: Sequence[FeatureRecord]) -> None:
        self.fit(records)

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict(self, features: FeatureRecord) -> dict[str, float]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path): ...

    @property
    @abstractmethod
    def classes_(self) -> Sequence[str]: ...
