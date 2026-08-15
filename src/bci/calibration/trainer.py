from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from bci.config import BCIConfig
from bci.domain import FeatureRecord
from bci.models.bayesian_latent import BayesianLatentDecoder


@dataclass
class CalibrationTrainer:
    config: BCIConfig
    decoder: BayesianLatentDecoder
    history: list[dict] = field(default_factory=list)

    def fit_batches(self, records: Sequence[FeatureRecord], artifact_dir: Path | None = None) -> BayesianLatentDecoder:
        accumulated: list[FeatureRecord] = []
        batch = max(1, self.config.calibration.batch_size_trials)
        for idx, record in enumerate(records, start=1):
            accumulated.append(record)
            if idx % batch != 0 and idx != len(records):
                continue
            labels = {r.label for r in accumulated}
            enough = all(
                sum(r.label == label for r in accumulated) >= self.config.calibration.minimum_trials_per_class_before_fit
                for label in labels
            )
            if not enough or len(labels) < 2:
                continue
            self.decoder.update(accumulated)
            snapshot = {
                "model_version": self.decoder.model_version,
                "n_records": len(accumulated),
                "record_ids": ";".join(r.trial_id for r in accumulated),
            }
            self.history.append(snapshot)
            if artifact_dir is not None:
                self.decoder.save(artifact_dir / f"model_v{self.decoder.model_version:03d}.pkl")
        return self.decoder
