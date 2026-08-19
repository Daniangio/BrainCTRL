from __future__ import annotations

from typing import Any

import numpy as np

from bci.adaptation.base import OnlineAdaptor
from bci.config import BCIConfig
from bci.domain import OnlineObservation
from bci.models.base import Decoder
from bci.models.riemannian import RiemannianMDMDecoder


class RiemannianPrototypeAdaptor(OnlineAdaptor):
    def __init__(self, config: BCIConfig):
        self.config = config
        self._current_label: str | None = None
        self._label_since: float | None = None
        self._last_update_time = -1.0e12

    def reset(self) -> None:
        self._current_label = None
        self._label_since = None
        self._last_update_time = -1.0e12

    def update(self, observation: OnlineObservation, decoder: Decoder) -> dict[str, Any]:
        row = self._base_row(observation)
        if not self.config.adaptation.enabled:
            row.update({"accepted": False, "reason": "disabled"})
            return row
        if self.config.adaptation.type != "riemannian_prototype":
            row.update({"accepted": False, "reason": "unsupported_adaptor"})
            return row
        if not isinstance(decoder, RiemannianMDMDecoder):
            row.update({"accepted": False, "reason": "unsupported_decoder"})
            return row
        if observation.phase.value not in set(self.config.adaptation.allowed_phases):
            row.update({"accepted": False, "reason": "phase_not_allowed"})
            return row
        if observation.quality is not None and observation.quality.score < self.config.quality.adaptation_min_quality:
            row.update({"accepted": False, "reason": "low_quality"})
            return row
        if observation.decision is None or observation.decision.command in {"", self.config.commands.reject_command}:
            self._track_label(None, observation.window_end)
            row.update({"accepted": False, "reason": "no_active_decision"})
            return row
        label = observation.decision.command
        if observation.decision.confidence < self.config.adaptation.min_temporal_posterior:
            self._track_label(label, observation.window_end)
            row.update({"accepted": False, "reason": "low_temporal_posterior"})
            return row
        margin = self._margin(observation.decision.probabilities)
        row["margin"] = margin
        if margin < self.config.adaptation.min_margin:
            self._track_label(label, observation.window_end)
            row.update({"accepted": False, "reason": "low_margin"})
            return row
        dwell = self._track_label(label, observation.window_end)
        row["dwell_seconds"] = dwell
        if dwell < self.config.adaptation.min_dwell_seconds:
            row.update({"accepted": False, "reason": "insufficient_dwell"})
            return row
        min_interval = 1.0 / max(self.config.adaptation.max_updates_per_second, 1.0e-9)
        if observation.window_end - self._last_update_time < min_interval:
            row.update({"accepted": False, "reason": "update_rate_limited"})
            return row
        matrices = observation.feature.covariance_matrices
        if matrices is None:
            row.update({"accepted": False, "reason": "missing_covariance"})
            return row
        accepted = decoder.adapt_prototype(
            label=label,
            matrices=np.asarray(matrices, dtype=float),
            eta=self.config.adaptation.eta,
            anchor_gamma=self.config.adaptation.anchor_gamma,
            max_anchor_distance=self.config.adaptation.max_anchor_distance,
        )
        if accepted:
            self._last_update_time = observation.window_end
            row.update({"accepted": True, "reason": "updated", "model_version_after": decoder.model_version})
        else:
            row.update({"accepted": False, "reason": "decoder_rejected"})
        return row

    def _base_row(self, observation: OnlineObservation) -> dict[str, Any]:
        return {
            "timestamp": observation.window_end,
            "window_id": observation.window_id,
            "phase": observation.phase.value,
            "label": observation.decision.command if observation.decision is not None else None,
            "confidence": observation.decision.confidence if observation.decision is not None else None,
            "quality_score": observation.quality.score if observation.quality is not None else None,
            "model_version_before": observation.model_version,
            "model_version_after": observation.model_version,
            "margin": None,
            "dwell_seconds": None,
        }

    def _track_label(self, label: str | None, timestamp: float) -> float:
        if label is None:
            self._current_label = None
            self._label_since = None
            return 0.0
        if label != self._current_label:
            self._current_label = label
            self._label_since = timestamp
            return 0.0
        if self._label_since is None:
            self._label_since = timestamp
            return 0.0
        return max(0.0, timestamp - self._label_since)

    def _margin(self, probabilities: dict[str, float]) -> float:
        values = sorted(probabilities.values(), reverse=True)
        if len(values) < 2:
            return values[0] if values else 0.0
        return float(values[0] - values[1])
