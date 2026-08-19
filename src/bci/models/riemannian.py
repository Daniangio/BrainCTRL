from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from pyriemann.classification import MDM
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.geodesic import geodesic_riemann
from scipy.special import log_softmax, softmax

from bci.config import BCIConfig
from bci.domain import DecoderDiagnostics, FeatureRecord
from bci.models.base import Decoder


class RiemannianMDMDecoder(Decoder):
    def __init__(self, config: BCIConfig):
        self.config = config
        self.model_version = 0
        self._classes: list[str] = []
        self._mdms: list[MDM] = []
        self._anchor_covmeans: list[np.ndarray] = []

    @property
    def classes_(self) -> Sequence[str]:
        return self._classes

    def fit(self, records: Sequence[FeatureRecord]) -> None:
        if not records:
            raise ValueError("RiemannianMDMDecoder requires at least one covariance feature")
        x = np.asarray([self._matrices_from_record(record) for record in records], dtype=float)
        y = np.asarray([record.label for record in records])
        self._mdms = []
        for band_idx in range(x.shape[1]):
            mdm = MDM(metric=self.config.model.riemannian_metric)
            mdm.fit(x[:, band_idx], y)
            self._mdms.append(mdm)
        self._classes = [str(label) for label in self._mdms[0].classes_]
        self._anchor_covmeans = [np.asarray(mdm.covmeans_, dtype=float).copy() for mdm in self._mdms]
        self.model_version += 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_version == 0:
            raise RuntimeError("RiemannianMDMDecoder is not fitted")
        matrices = np.asarray(X, dtype=float)
        if matrices.ndim == 2:
            matrices = matrices[None, None, :, :]
        elif matrices.ndim == 3:
            matrices = matrices[:, None, :, :]
        return np.asarray([self._probabilities_from_matrices(sample) for sample in matrices])

    def predict(self, features: FeatureRecord) -> dict[str, float]:
        if self.model_version == 0:
            raise RuntimeError("RiemannianMDMDecoder is not fitted")
        probs = self._probabilities_from_matrices(self._matrices_from_record(features))
        return dict(zip(self._classes, map(float, probs)))

    def diagnostics(self, records: Sequence[FeatureRecord] | None = None) -> DecoderDiagnostics:
        centers = {
            str(label): np.asarray(center, dtype=float).copy()
            for label, center in zip(self._classes, self._mdms[0].covmeans_)
        }
        separation: dict[str, float] = {}
        for band_idx, mdm in enumerate(self._mdms):
            for i, left in enumerate(self._classes):
                for j, right in enumerate(self._classes):
                    if j <= i:
                        continue
                    value = float(distance_riemann(mdm.covmeans_[i], mdm.covmeans_[j]))
                    separation[f"band{band_idx + 1}:{left}_vs_{right}"] = value
                    if band_idx == 0:
                        separation[f"{left}_vs_{right}"] = value
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

    def adapt_prototype(self, label: str, matrices: np.ndarray, eta: float, anchor_gamma: float, max_anchor_distance: float) -> bool:
        if self.model_version == 0 or label not in self._classes:
            return False
        if not self._anchor_covmeans:
            self._anchor_covmeans = [np.asarray(mdm.covmeans_, dtype=float).copy() for mdm in self._mdms]
        class_idx = self._classes.index(label)
        matrices = np.asarray(matrices, dtype=float)
        if matrices.ndim != 3 or matrices.shape[0] < len(self._mdms):
            return False
        candidates: list[np.ndarray] = []
        for band_idx, mdm in enumerate(self._mdms):
            current = np.asarray(mdm.covmeans_[class_idx], dtype=float)
            candidate = geodesic_riemann(current, matrices[band_idx], alpha=eta)
            anchor = self._anchor_covmeans[band_idx][class_idx]
            if anchor_gamma > 0.0:
                candidate = geodesic_riemann(candidate, anchor, alpha=anchor_gamma * eta)
            if distance_riemann(candidate, anchor) > max_anchor_distance:
                return False
            candidates.append(candidate)
        for candidate, mdm in zip(candidates, self._mdms):
            mdm.covmeans_[class_idx] = candidate
        self.model_version += 1
        return True

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "RiemannianMDMDecoder":
        with path.open("rb") as f:
            return pickle.load(f)

    def _matrices_from_record(self, record: FeatureRecord) -> np.ndarray:
        if record.covariance_matrices is None:
            raise ValueError("RiemannianMDMDecoder requires FeatureRecord.covariance_matrices")
        matrices = np.asarray(record.covariance_matrices, dtype=float)
        if matrices.ndim != 3 or matrices.shape[0] < 1:
            raise ValueError("covariance_matrices must have shape (n_bands, n_channels, n_channels)")
        return matrices

    def _probabilities_from_matrices(self, matrices: np.ndarray) -> np.ndarray:
        temperature = max(self.config.model.probability_temperature, 1.0e-6)
        band_log_probs: list[np.ndarray] = []
        for matrix, mdm in zip(matrices, self._mdms):
            distances = np.asarray([distance_riemann(matrix, center) for center in mdm.covmeans_], dtype=float)
            band_log_probs.append(log_softmax(-(distances**2) / temperature))
        if not band_log_probs:
            return np.full(len(self._classes), 1.0 / len(self._classes))
        return softmax(np.mean(np.asarray(band_log_probs), axis=0))
