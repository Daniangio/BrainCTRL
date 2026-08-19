from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.covariance import LedoitWolf, OAS

from bci.config import BCIConfig
from bci.domain import FeatureRecord, TrialRecord
from bci.features.base import FeatureExtractor
from bci.utils.hashing import stable_hash


class CovarianceFeatureExtractor(FeatureExtractor):
    version = "covariance_v1"

    def __init__(self, config: BCIConfig):
        self.config = config
        self.config_hash = stable_hash(
            {
                "version": self.version,
                "features": config.features.model_dump(mode="json"),
                "channels": config.channels.model_dump(mode="json"),
            }
        )

    def transform(self, trial: TrialRecord) -> FeatureRecord:
        data = np.asarray(trial.data, dtype=float)
        matrices: list[np.ndarray] = []
        values: list[float] = []
        names: list[str] = []
        band_names: list[str] = []
        for band in self._valid_bands(trial.sfreq):
            band_name = f"{band[0]:g}-{band[1]:g}Hz"
            filtered = self._filter_band(data, trial.sfreq, band)
            cov = self._estimate_covariance(filtered)
            matrices.append(cov)
            band_names.append(band_name)
            tri_i, tri_j = np.triu_indices(cov.shape[0])
            values.extend(cov[tri_i, tri_j].tolist())
            names.extend(f"{band_name}:{trial.ch_names[i]}:{trial.ch_names[j]}:cov" for i, j in zip(tri_i, tri_j))
        return FeatureRecord(
            trial_id=trial.trial_id,
            label=trial.command,
            split=trial.split or "unassigned",
            values=np.asarray(values, dtype=float),
            feature_names=names,
            frequency_scores={},
            provenance={
                "dataset": trial.dataset,
                "subject": trial.subject,
                "session": trial.session,
                "run": trial.run,
                "event_index": trial.event_index,
                "source_event_id": trial.source_event_id if trial.source_event_id is not None else trial.event_index,
                "native_label": trial.native_label,
                "start_time": trial.start_time,
                "end_time": trial.end_time,
            },
            config_hash=self.config_hash,
            spectral_freqs=self._spectral_freqs(data.shape[1], trial.sfreq),
            spectral_power=self._spectral_power(data),
            spectral_channel_names=list(trial.ch_names),
            covariance_matrices=np.asarray(matrices, dtype=float),
            covariance_band_names=band_names,
            representation_type="covariance",
        )

    def _valid_bands(self, sfreq: float) -> list[tuple[float, float]]:
        nyq = sfreq / 2.0
        bands: list[tuple[float, float]] = []
        for low, high in self.config.features.covariance.bands_hz:
            low = max(float(low), 0.0)
            high = min(float(high), nyq * 0.98)
            if high > low > 0.0:
                bands.append((low, high))
        return bands or [(1.0, nyq * 0.98)]

    def _filter_band(self, data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
        low, high = band
        nyq = sfreq / 2.0
        if low <= 0.0 and high >= nyq * 0.95:
            return data
        sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
        max_pad = 3 * (2 * sos.shape[0] + 1)
        if data.shape[1] > max_pad:
            return signal.sosfiltfilt(sos, data, axis=1)
        return signal.sosfilt(sos, data, axis=1)

    def _estimate_covariance(self, data: np.ndarray) -> np.ndarray:
        samples = np.asarray(data, dtype=float).T
        estimator = self.config.features.covariance.estimator
        if estimator == "oas" and samples.shape[0] > 1:
            cov = OAS(store_precision=False).fit(samples).covariance_
        elif estimator == "ledoit_wolf" and samples.shape[0] > 1:
            cov = LedoitWolf(store_precision=False).fit(samples).covariance_
        else:
            cov = np.cov(samples, rowvar=False, bias=False)
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        cov = (cov + cov.T) / 2.0
        cov += self.config.features.covariance.regularization * np.eye(cov.shape[0])
        if self.config.features.covariance.normalize == "trace":
            trace = float(np.trace(cov))
            if trace > 0.0:
                cov = cov / trace * cov.shape[0]
        return cov

    def _spectral_freqs(self, n_samples: int, sfreq: float) -> np.ndarray:
        return np.fft.rfftfreq(n_samples, d=1.0 / sfreq)

    def _spectral_power(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[1]
        window = np.hanning(n)
        spectrum = np.fft.rfft(data * window[None, :], axis=1)
        return (np.abs(spectrum) ** 2) / max(float(np.sum(window**2)), 1.0)
