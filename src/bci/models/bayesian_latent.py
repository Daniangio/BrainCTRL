from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

from bci.config import BCIConfig
from bci.domain import DecoderDiagnostics, FeatureRecord
from bci.models.base import Decoder


class BayesianLatentDecoder(Decoder):
    def __init__(self, config: BCIConfig):
        self.config = config
        self.model_version = 0
        self._classes: list[str] = []
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.latent_means_: dict[str, np.ndarray] = {}
        self.cov_: np.ndarray | None = None
        self.priors_: dict[str, float] = {}

    @property
    def classes_(self) -> Sequence[str]:
        return self._classes

    def fit(self, records: Sequence[FeatureRecord]) -> None:
        if not records:
            raise ValueError("cannot fit decoder with no records")
        X = np.vstack([r.values for r in records])
        y = np.asarray([r.label for r in records])
        self._classes = sorted(
            {str(label) for label in y},
            key=lambda c: ("LEFT", "RIGHT", "NONE").index(c) if c in {"LEFT", "RIGHT", "NONE"} else c,
        )
        self.x_mean_ = X.mean(axis=0) if self.config.model.standardize_features else np.zeros(X.shape[1])
        self.x_scale_ = X.std(axis=0) if self.config.model.standardize_features else np.ones(X.shape[1])
        self.x_scale_[self.x_scale_ < 1e-9] = 1.0
        Xs = (X - self.x_mean_) / self.x_scale_
        self.W_ = self._fit_projection(Xs, y)
        Z = Xs @ self.W_
        self.latent_means_ = {c: Z[y == c].mean(axis=0) for c in self._classes}
        centered = np.vstack([Z[y == c] - self.latent_means_[c] for c in self._classes if np.any(y == c)])
        if centered.shape[0] <= 1:
            cov = np.eye(Z.shape[1])
        else:
            cov = np.cov(centered, rowvar=False, bias=False)
            cov = np.atleast_2d(cov)
        reg = self.config.model.regularization
        shrink_target = np.eye(cov.shape[0]) * max(float(np.trace(cov) / cov.shape[0]), reg)
        self.cov_ = 0.8 * cov + 0.2 * shrink_target + reg * np.eye(cov.shape[0])
        counts = {c: int(np.sum(y == c)) for c in self._classes}
        total = float(len(y))
        self.priors_ = {c: counts[c] / total for c in self._classes}
        self.model_version += 1

    def _fit_projection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes = self._classes
        overall = X.mean(axis=0)
        d = X.shape[1]
        sw = np.zeros((d, d))
        sb = np.zeros((d, d))
        for c in classes:
            Xc = X[y == c]
            mean = Xc.mean(axis=0)
            centered = Xc - mean
            sw += centered.T @ centered
            diff = (mean - overall)[:, None]
            sb += Xc.shape[0] * (diff @ diff.T)
        reg = self.config.model.regularization
        diag = np.diag(np.diag(sw))
        sw = 0.8 * sw + 0.2 * diag + reg * np.eye(d)
        latent_dim = max(1, min(self.config.model.latent_dim, len(classes) - 1, d))
        if len(classes) == 2:
            means = [X[y == c].mean(axis=0) for c in classes]
            w = np.linalg.solve(sw, means[0] - means[1])
            W = w[:, None]
        else:
            vals, vecs = eigh(sb, sw)
            order = np.argsort(vals)[::-1][:latent_dim]
            W = vecs[:, order]
        norms = np.linalg.norm(W, axis=0)
        norms[norms < 1e-12] = 1.0
        return W / norms

    @staticmethod
    def fisher_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        z = X @ w
        classes = np.unique(y)
        overall = z.mean()
        between = sum(np.sum(y == c) * float((z[y == c].mean() - overall) ** 2) for c in classes)
        within = sum(float(np.sum((z[y == c] - z[y == c].mean()) ** 2)) for c in classes)
        return between / max(within, 1e-12)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_scale_ is None or self.W_ is None:
            raise RuntimeError("decoder is not fitted")
        return ((np.atleast_2d(X) - self.x_mean_) / self.x_scale_) @ self.W_

    def transform_latent(self, X: np.ndarray) -> np.ndarray:
        return self._transform(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.cov_ is None:
            raise RuntimeError("decoder is not fitted")
        Z = self._transform(X)
        inv_cov = np.linalg.inv(self.cov_)
        _, logdet = np.linalg.slogdet(self.cov_)
        rows = []
        for z in Z:
            logs = []
            for c in self._classes:
                diff = z - self.latent_means_[c]
                quad = float(diff.T @ inv_cov @ diff)
                logs.append(np.log(self.priors_[c]) - 0.5 * (quad + logdet))
            logs = np.asarray(logs)
            rows.append(np.exp(logs - logsumexp(logs)))
        return np.vstack(rows)

    def predict(self, features: FeatureRecord) -> dict[str, float]:
        probs = self.predict_proba(features.values)[0]
        return dict(zip(self._classes, map(float, probs)))

    def diagnostics(self, records: Sequence[FeatureRecord] | None = None) -> DecoderDiagnostics:
        latent_points = None
        latent_labels = None
        if records:
            latent_points = self.transform_latent(np.vstack([r.values for r in records]))
            latent_labels = [r.label for r in records]
        covariances = {c: np.asarray(self.cov_).copy() for c in self._classes} if self.cov_ is not None else {}
        separation: dict[str, float] = {}
        if self.cov_ is not None and len(self._classes) >= 2:
            inv_cov = np.linalg.inv(self.cov_)
            for i, a in enumerate(self._classes):
                for b in self._classes[i + 1 :]:
                    diff = self.latent_means_[a] - self.latent_means_[b]
                    separation[f"{a}_vs_{b}"] = float(np.sqrt(max(diff.T @ inv_cov @ diff, 0.0)))
        return DecoderDiagnostics(
            model_version=self.model_version,
            classes=list(self._classes),
            latent_dim=0 if self.W_ is None else int(self.W_.shape[1]),
            latent_points=latent_points,
            latent_labels=latent_labels,
            class_centers={k: v.copy() for k, v in self.latent_means_.items()},
            class_covariances=covariances,
            separation=separation,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "BayesianLatentDecoder":
        with path.open("rb") as f:
            return pickle.load(f)
