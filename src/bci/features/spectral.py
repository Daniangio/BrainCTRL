from __future__ import annotations

import numpy as np

from bci.config import BCIConfig
from bci.domain import FeatureRecord, TrialRecord
from bci.features.base import FeatureExtractor
from bci.utils.hashing import stable_hash


class SpectralFeatureExtractor(FeatureExtractor):
    version = "spectral_relative_power_v1"

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
        n = data.shape[1]
        window = np.hanning(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / trial.sfreq)
        spectrum = np.fft.rfft(data * window[None, :], axis=1)
        power = (np.abs(spectrum) ** 2) / max(float(np.sum(window**2)), 1.0)
        log_power = np.log(power + self.config.features.log_epsilon)

        values: list[float] = []
        names: list[str] = []
        omitted: list[dict[str, object]] = []
        scores = {command: 0.0 for command in self.command_frequencies}
        counts = {command: 0 for command in self.command_frequencies}
        upper_band = self._effective_upper_band(trial.sfreq)
        for ch_idx, ch_name in enumerate(trial.ch_names):
            for command, base_freq in self.command_frequencies.items():
                for harmonic in self.config.features.harmonics:
                    target = base_freq * harmonic
                    if target >= trial.sfreq / 2.0:
                        omitted.append({"command": command, "frequency": target, "harmonic": harmonic, "reason": "above_nyquist"})
                        continue
                    if upper_band is not None and target > upper_band:
                        omitted.append({"command": command, "frequency": target, "harmonic": harmonic, "reason": "above_preprocessing_band"})
                        continue
                    value = self._local_log_snr(freqs, log_power[ch_idx], target)
                    names.append(f"{ch_name}:{base_freq:g}Hz:h{harmonic}:local_log_snr")
                    values.append(value)
                    scores[command] += value
                    counts[command] += 1
        for command, count in counts.items():
            if count:
                scores[command] /= count
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
            spectral_freqs=freqs.copy(),
            log_power=log_power.mean(axis=0),
            omitted_harmonics=omitted,
        )

    def _effective_upper_band(self, sfreq: float) -> float | None:
        if self.config.preprocessing.bandpass_hz is None:
            return None
        return min(float(self.config.preprocessing.bandpass_hz[1]), sfreq / 2.0)

    def _local_log_snr(self, freqs: np.ndarray, log_power: np.ndarray, target: float) -> float:
        half = self.config.features.local_band_half_width_hz
        inner = self.config.features.neighbor_inner_gap_hz
        outer = self.config.features.neighbor_outer_width_hz
        target_mask = np.abs(freqs - target) <= half
        if not np.any(target_mask):
            target_idx = int(np.argmin(np.abs(freqs - target)))
            target_value = float(log_power[target_idx])
        else:
            target_value = float(np.mean(log_power[target_mask]))
        neighbor_mask = (np.abs(freqs - target) >= inner) & (np.abs(freqs - target) <= outer)
        if not np.any(neighbor_mask):
            return target_value
        return target_value - float(np.mean(log_power[neighbor_mask]))
