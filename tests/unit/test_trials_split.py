from __future__ import annotations

import numpy as np

from bci.config import load_config
from bci.domain import TrialRecord
from bci.splitting.chronological import ChronologicalTrialSplit, apply_split


def make_trial(i: int, command: str) -> TrialRecord:
    return TrialRecord(
        trial_id=f"t{i}",
        dataset="D",
        subject=1,
        session="0",
        run="0",
        event_index=i,
        native_label="13" if command == "LEFT" else "21",
        command=command,
        start_time=float(i),
        end_time=float(i) + 1.5,
        sfreq=256.0,
        ch_names=["Oz"],
        data=np.zeros((1, 384)),
    )


def test_trial_split_has_no_provenance_overlap():
    config = load_config("configs/kalunga_v0.yaml")
    trials = [make_trial(i, "LEFT" if i % 2 == 0 else "RIGHT") for i in range(12)]
    manifest = ChronologicalTrialSplit(config).assign(trials)
    split = apply_split(trials, manifest)
    groups = {}
    for trial in split:
        assert trial.provenance_key not in groups
        groups[trial.provenance_key] = trial.split
    assert {"calibration", "validation", "test"} <= {t.split for t in split}
