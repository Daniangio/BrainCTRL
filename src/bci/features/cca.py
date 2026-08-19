from __future__ import annotations

import numpy as np
from scipy import signal

from bci.config import BCIConfig
from bci.domain import FeatureRecord, TrialRecord
from bci.features.base import FeatureExtractor
from bci.utils.hashing import stable_hash


class FBCCAFeatureExtractor(FeatureExtractor):
    version = "fbcca_v1"

    def __init__(self, config: BCIConfig):
        self.config = config
        self.command_frequencies = config.stimulus_frequencies
        self.config_hash = stable_hash(
            {
                "version": self.version,
                "features": config.features.model_dump(mode="json"),
                "commands": config.commands.model_dump(mode="json"),
                "channels": config.channels.model_dump(mode="json"),
            }
        )

    def transform(self, trial: TrialRecord) -> FeatureRecord:
        data = np.asarray(trial.data, dtype=float)
        bands = self._valid_bands(trial.sfreq)
        band_weights = self._band_weights(len(bands))
        values: list[float] = []
        names: list[str] = []
        scores = {command: 0.0 for command in self.command_frequencies}
        omitted: list[dict[str, object]] = []
        for band_idx, band in enumerate(bands):
            filtered = self._filter_band(data, trial.sfreq, band)
            weight = band_weights[band_idx]
            for command, base_freq in self.command_frequencies.items():
                harmonics = [
                    harmonic
                    for harmonic in self.config.features.fbcca.harmonics
                    if base_freq * harmonic < trial.sfreq / 2.0
                ]
                for harmonic in self.config.features.fbcca.harmonics:
                    if base_freq * harmonic >= trial.sfreq / 2.0:
                        omitted.append(
                            {
                                "command": command,
                                "frequency": base_freq * harmonic,
                                "harmonic": harmonic,
                                "reason": "above_nyquist",
                            }
                        )
                if not harmonics:
                    rho = 0.0
                else:
                    refs = self._references(base_freq, harmonics, trial.sfreq, data.shape[1])
                    rho = self._first_canonical_correlation(filtered, refs)
                score = float(weight * rho * rho)
                values.append(score)
                names.append(f"fb{band_idx + 1}:{band[0]:g}-{band[1]:g}Hz:{command}:rho2_weighted")
                scores[command] += score
        return FeatureRecord(
            trial_id=trial.trial_id,
            label=trial.command,
            split=trial.split or "unassigned",
            values=np.asarray(values, dtype=float),
            feature_names=names,
            frequency_scores=scores,
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
            spectral_power=self._spectral_power(data, trial.sfreq),
            spectral_channel_names=list(trial.ch_names),
            omitted_harmonics=omitted,
        )

    def _valid_bands(self, sfreq: float) -> list[tuple[float, float]]:
        nyq = sfreq / 2.0
        bands: list[tuple[float, float]] = []
        for low, high in self.config.features.fbcca.bands_hz:
            high = min(float(high), nyq * 0.98)
            low = max(float(low), 0.0)
            if high > low > 0.0:
                bands.append((low, high))
        return bands or [(1.0, nyq * 0.98)]

    def _band_weights(self, n_bands: int) -> np.ndarray:
        idx = np.arange(1, n_bands + 1, dtype=float)
        weights = idx ** -1.25 + 0.25
        return weights / np.sum(weights)

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

    def _references(self, frequency: float, harmonics: list[int], sfreq: float, n_samples: int) -> np.ndarray:
        times = np.arange(n_samples, dtype=float) / sfreq
        refs: list[np.ndarray] = []
        for harmonic in harmonics:
            phase = 2.0 * np.pi * frequency * harmonic * times
            refs.append(np.sin(phase))
            refs.append(np.cos(phase))
        return np.asarray(refs, dtype=float)

    def _first_canonical_correlation(self, data: np.ndarray, refs: np.ndarray) -> float:
        x = np.asarray(data, dtype=float).T
        y = np.asarray(refs, dtype=float).T
        x = x - np.mean(x, axis=0, keepdims=True)
        y = y - np.mean(y, axis=0, keepdims=True)
        denom = max(x.shape[0] - 1, 1)
        reg = self.config.features.fbcca.regularization
        sxx = (x.T @ x) / denom + reg * np.eye(x.shape[1])
        syy = (y.T @ y) / denom + reg * np.eye(y.shape[1])
        sxy = (x.T @ y) / denom
        matrix = np.linalg.pinv(sxx) @ sxy @ np.linalg.pinv(syy) @ sxy.T
        eigvals = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
        return float(np.sqrt(np.clip(np.max(eigvals), 0.0, 1.0)))

    def _spectral_freqs(self, n_samples: int, sfreq: float) -> np.ndarray:
        return np.fft.rfftfreq(n_samples, d=1.0 / sfreq)

    def _spectral_power(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        n = data.shape[1]
        window = np.hanning(n)
        spectrum = np.fft.rfft(data * window[None, :], axis=1)
        return (np.abs(spectrum) ** 2) / max(float(np.sum(window**2)), 1.0)
