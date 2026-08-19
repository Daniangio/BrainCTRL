from __future__ import annotations

import math

import numpy as np

from bci.config import BCIConfig
from bci.domain import SignalQuality


class SignalQualityEstimator:
    def __init__(self, config: BCIConfig):
        self.config = config
        self._observed_seconds = 0.0
        self._log_var_mean: np.ndarray | None = None
        self._log_var_m2: np.ndarray | None = None

    def reset(self) -> None:
        self._observed_seconds = 0.0
        self._log_var_mean = None
        self._log_var_m2 = None

    def estimate(self, data: np.ndarray, sfreq: float, ch_names: list[str]) -> SignalQuality:
        x = np.asarray(data, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            return SignalQuality(
                score=0.0,
                flags=["empty_window"],
                per_channel={name: 0.0 for name in ch_names},
                metrics={"finite_fraction": 0.0},
                history_ready=False,
            )

        finite = np.isfinite(x)
        finite_fraction = float(finite.mean())
        clean = np.where(finite, x, 0.0)
        variances = np.var(clean, axis=1)
        amplitudes = np.ptp(clean, axis=1)
        abs_clean = np.abs(clean)
        clip_level = np.percentile(abs_clean, 99.9) if abs_clean.size else 0.0
        clipping_fraction = float(np.mean(abs_clean >= clip_level)) if clip_level > 0 else 0.0
        flat = (variances < 1.0e-12) | (amplitudes < 1.0e-9)
        hf_ratio, line_ratio = self._spectral_ratios(clean, sfreq)

        history_ready = self._observed_seconds >= self.config.quality.warmup_seconds
        z_scores = self._variance_z_scores(variances) if history_ready else np.zeros_like(variances)
        bad_by_variance = np.abs(z_scores) > 4.0
        bad_channels = flat | bad_by_variance | (~finite.all(axis=1))
        bad_fraction = float(np.mean(bad_channels))

        flags: list[str] = []
        if finite_fraction < 1.0:
            flags.append("non_finite")
        if clipping_fraction > 0.02:
            flags.append("clipping")
        if np.any(flat):
            flags.append("flat_channel")
        if history_ready and np.any(bad_by_variance):
            flags.append("variance_outlier")
        if hf_ratio > 0.45:
            flags.append("high_frequency_noise")
        if line_ratio > 0.35:
            flags.append("line_noise")
        if bad_fraction > self.config.quality.max_bad_channels_fraction:
            flags.append("too_many_bad_channels")

        score = 1.0
        score *= finite_fraction
        score *= max(0.0, 1.0 - bad_fraction)
        score *= max(0.0, 1.0 - 3.0 * clipping_fraction)
        score *= max(0.0, 1.0 - max(0.0, hf_ratio - 0.25))
        score *= max(0.0, 1.0 - max(0.0, line_ratio - 0.20))
        score = float(np.clip(score, 0.0, 1.0))
        if score < self.config.quality.hard_reject_threshold and "hard_reject" not in flags:
            flags.append("hard_reject")

        self._update_history(variances, x.shape[1] / sfreq)
        per_channel = {
            ch_names[i] if i < len(ch_names) else f"ch{i}": float(np.clip(1.0 - float(bad_channels[i]), 0.0, 1.0))
            for i in range(x.shape[0])
        }
        return SignalQuality(
            score=score,
            flags=flags,
            per_channel=per_channel,
            metrics={
                "finite_fraction": finite_fraction,
                "bad_channels_fraction": bad_fraction,
                "clipping_fraction": clipping_fraction,
                "high_frequency_ratio": hf_ratio,
                "line_noise_ratio": line_ratio,
                "max_variance_z": float(np.max(np.abs(z_scores))) if z_scores.size else 0.0,
            },
            history_ready=history_ready,
        )

    def _spectral_ratios(self, data: np.ndarray, sfreq: float) -> tuple[float, float]:
        if data.shape[1] < 4:
            return 0.0, 0.0
        freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
        power = np.abs(np.fft.rfft(data, axis=1)) ** 2
        total_band = (freqs >= 1.0) & (freqs <= min(0.49 * sfreq, 80.0))
        total = float(np.sum(power[:, total_band]))
        if total <= 0.0 or not math.isfinite(total):
            return 0.0, 0.0
        hf_start = min(self.config.quality.high_frequency_start_hz, 0.49 * sfreq)
        hf_band = freqs >= hf_start
        line = self.config.quality.line_frequency_hz
        line_band = (freqs >= line - 1.0) & (freqs <= line + 1.0)
        return float(np.sum(power[:, hf_band]) / total), float(np.sum(power[:, line_band]) / total)

    def _variance_z_scores(self, variances: np.ndarray) -> np.ndarray:
        if self._log_var_mean is None or self._log_var_m2 is None:
            return np.zeros_like(variances)
        log_var = np.log(np.maximum(variances, 1.0e-18))
        scale = np.sqrt(np.maximum(self._log_var_m2, 1.0e-6))
        return (log_var - self._log_var_mean) / scale

    def _update_history(self, variances: np.ndarray, seconds: float) -> None:
        log_var = np.log(np.maximum(variances, 1.0e-18))
        if self._log_var_mean is None or self._log_var_m2 is None or self._log_var_mean.shape != log_var.shape:
            self._log_var_mean = log_var.copy()
            self._log_var_m2 = np.full_like(log_var, 1.0)
            self._observed_seconds += seconds
            return
        alpha = 1.0 - math.exp(-seconds / max(self.config.quality.history_tau_seconds, 1.0e-6))
        delta = log_var - self._log_var_mean
        self._log_var_mean += alpha * delta
        self._log_var_m2 = (1.0 - alpha) * (self._log_var_m2 + alpha * delta * delta)
        self._observed_seconds += seconds
