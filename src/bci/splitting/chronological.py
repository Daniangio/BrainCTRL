from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Sequence

from bci.config import BCIConfig
from bci.domain import TrialRecord
from bci.splitting.base import SplitPolicy


class ChronologicalTrialSplit(SplitPolicy):
    def __init__(self, config: BCIConfig):
        self.config = config

    def assign(self, trials: Sequence[TrialRecord]) -> dict[str, str]:
        if self.config.split.stratify_if_possible:
            by_label: dict[str, list[TrialRecord]] = defaultdict(list)
            for trial in trials:
                by_label[trial.command].append(trial)
            manifest: dict[str, str] = {}
            for label_trials in by_label.values():
                manifest.update(self._assign_ordered(label_trials))
            return manifest
        return self._assign_ordered(list(trials))

    def _assign_ordered(self, trials: Sequence[TrialRecord]) -> dict[str, str]:
        ordered = sorted(trials, key=lambda t: (t.subject, t.session, t.run, t.start_time, t.event_index))
        n = len(ordered)
        n_cal = int(round(n * self.config.split.calibration_fraction))
        n_val = int(round(n * self.config.split.validation_fraction))
        if n >= 3:
            n_cal = max(1, min(n - 2, n_cal))
            n_val = max(1, min(n - n_cal - 1, n_val))
        manifest = {}
        for idx, trial in enumerate(ordered):
            if idx < n_cal:
                split = "calibration"
            elif idx < n_cal + n_val:
                split = "validation"
            else:
                split = "test"
            manifest[trial.trial_id] = split
        return manifest


def apply_split(trials: Sequence[TrialRecord], manifest: dict[str, str]) -> list[TrialRecord]:
    return [replace(t, split=manifest[t.trial_id]) for t in trials]
