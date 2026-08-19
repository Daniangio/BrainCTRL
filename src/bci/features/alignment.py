from __future__ import annotations

from dataclasses import replace

import numpy as np

from bci.config import BCIConfig
from bci.domain import FeatureRecord, SignalQuality


class EuclideanAlignment:
    def __init__(self, config: BCIConfig):
        self.config = config
        self.version = 0
        self.frozen = False
        self.observed_seconds = 0.0
        self._reference: np.ndarray | None = None
        self._inverse_sqrt: np.ndarray | None = None
        self._n_updates = 0

    def reset(self) -> None:
        self.version = 0
        self.frozen = False
        self.observed_seconds = 0.0
        self._reference = None
        self._inverse_sqrt = None
        self._n_updates = 0

    def update(self, feature: FeatureRecord, quality: SignalQuality | None = None) -> None:
        if not self._enabled() or feature.covariance_matrices is None or self.frozen:
            return
        if quality is not None and quality.score < self.config.alignment.min_quality:
            return
        matrices = np.asarray(feature.covariance_matrices, dtype=float)
        if matrices.ndim != 3:
            return
        seconds = self._duration(feature)
        if self._reference is None or self._reference.shape != matrices.shape:
            self._reference = matrices.copy()
        elif self.config.alignment.mode == "slow_online":
            alpha = 1.0 - np.exp(-seconds / max(self.config.alignment.update_tau_seconds, 1.0e-6))
            self._reference = (1.0 - alpha) * self._reference + alpha * matrices
        else:
            self._reference = (self._reference * self._n_updates + matrices) / (self._n_updates + 1)
        self._n_updates += 1
        self.observed_seconds += seconds
        self._inverse_sqrt = np.asarray([self._matrix_inverse_sqrt(matrix) for matrix in self._reference])
        self.version += 1
        if self.config.alignment.mode == "warmup_freeze" and self.observed_seconds >= self.config.alignment.warmup_seconds:
            self.frozen = True

    def transform(self, feature: FeatureRecord) -> FeatureRecord:
        if not self._enabled() or feature.covariance_matrices is None or self._inverse_sqrt is None:
            return feature
        matrices = np.asarray(feature.covariance_matrices, dtype=float)
        if matrices.shape != self._inverse_sqrt.shape:
            return feature
        aligned = np.asarray([w @ cov @ w for w, cov in zip(self._inverse_sqrt, matrices)], dtype=float)
        aligned = np.asarray([self._regularize((matrix + matrix.T) / 2.0) for matrix in aligned], dtype=float)
        values = self._vectorize(aligned)
        provenance = dict(feature.provenance)
        provenance.update(
            {
                "alignment_type": self.config.alignment.type,
                "alignment_version": self.version,
                "alignment_frozen": self.frozen,
            }
        )
        return replace(
            feature,
            values=values,
            covariance_matrices=aligned,
            alignment_version=self.version,
            provenance=provenance,
        )

    def update_transform(self, feature: FeatureRecord, quality: SignalQuality | None = None) -> FeatureRecord:
        self.update(feature, quality)
        return self.transform(feature)

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": self._enabled(),
            "type": self.config.alignment.type,
            "mode": self.config.alignment.mode,
            "version": self.version,
            "frozen": self.frozen,
            "observed_seconds": self.observed_seconds,
            "n_updates": self._n_updates,
        }

    def _enabled(self) -> bool:
        return self.config.alignment.enabled and self.config.alignment.type == "euclidean"

    def _duration(self, feature: FeatureRecord) -> float:
        start = feature.provenance.get("start_time")
        end = feature.provenance.get("end_time")
        if start is None or end is None:
            return self.config.trials.window_seconds
        return max(float(end) - float(start), 0.0)

    def _matrix_inverse_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        matrix = self._regularize((matrix + matrix.T) / 2.0)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, self.config.alignment.regularization)
        return (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

    def _regularize(self, matrix: np.ndarray) -> np.ndarray:
        return matrix + self.config.alignment.regularization * np.eye(matrix.shape[0])

    def _vectorize(self, matrices: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for matrix in matrices:
            tri_i, tri_j = np.triu_indices(matrix.shape[0])
            values.extend(matrix[tri_i, tri_j].tolist())
        return np.asarray(values, dtype=float)
