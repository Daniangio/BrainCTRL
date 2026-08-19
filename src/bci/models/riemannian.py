from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from pyriemann.classification import MDM
from pyriemann.geometry.distance import distance_riemann
from scipy.special import softmax

from bci.config import BCIConfig
from bci.domain import DecoderDiagnostics, FeatureRecord
from bci.models.base import Decoder


class RiemannianMDMDecoder(Decoder):
    def __init__(self, config: BCIConfig):
        self.config = config
        self.model_version = 0
        self._classes: list[str] = []
        self._mdm = MDM(metric=config.model.riemannian_metric)

    @property
    def classes_(self) -> Sequence[str]:
        return self._classes

    def fit(self, records: Sequence[FeatureRecord]) -> None:
        if not records:
            raise ValueError("RiemannianMDMDecoder requires at least one covariance feature")
        x = np.asarray([self._matrix_from_record(record) for record in records], dtype=float)
        y = np.asarray([record.label for record in records])
        self._mdm = MDM(metric=self.config.model.riemannian_metric)
        self._mdm.fit(x, y)
        self._classes = [str(label) for label in self._mdm.classes_]
        self.model_version += 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_version == 0:
            raise RuntimeError("RiemannianMDMDecoder is not fitted")
        matrices = np.asarray(X, dtype=float)
        if matrices.ndim == 2:
            matrices = matrices[None, :, :]
        return np.asarray([self._probabilities_from_matrix(matrix) for matrix in matrices])

    def predict(self, features: FeatureRecord) -> dict[str, float]:
        if self.model_version == 0:
            raise RuntimeError("RiemannianMDMDecoder is not fitted")
        probs = self._probabilities_from_matrix(self._matrix_from_record(features))
        return dict(zip(self._classes, map(float, probs)))

    def diagnostics(self, records: Sequence[FeatureRecord] | None = None) -> DecoderDiagnostics:
        centers = {
            str(label): np.asarray(center, dtype=float).copy()
            for label, center in zip(self._classes, self._mdm.covmeans_)
        }
        separation: dict[str, float] = {}
        for i, left in enumerate(self._classes):
            for j, right in enumerate(self._classes):
                if j <= i:
                    continue
                separation[f"{left}_vs_{right}"] = float(distance_riemann(centers[left], centers[right]))
        return DecoderDiagnostics(
            model_version=self.model_version,
            classes=list(self._classes),
            latent_dim=0,
            latent_points=None,
            latent_labels=None,
            class_centers=centers,
            class_covariances={},
            separation=separation,
        )

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "RiemannianMDMDecoder":
        with path.open("rb") as f:
            return pickle.load(f)

    def _matrix_from_record(self, record: FeatureRecord) -> np.ndarray:
        if record.covariance_matrices is None:
            raise ValueError("RiemannianMDMDecoder requires FeatureRecord.covariance_matrices")
        matrices = np.asarray(record.covariance_matrices, dtype=float)
        if matrices.ndim != 3 or matrices.shape[0] < 1:
            raise ValueError("covariance_matrices must have shape (n_bands, n_channels, n_channels)")
        return matrices[0]

    def _probabilities_from_matrix(self, matrix: np.ndarray) -> np.ndarray:
        distances = np.asarray([distance_riemann(matrix, center) for center in self._mdm.covmeans_], dtype=float)
        temperature = max(self.config.model.probability_temperature, 1.0e-6)
        return softmax(-(distances**2) / temperature)
