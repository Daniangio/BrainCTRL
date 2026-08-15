from __future__ import annotations

from typing import Sequence

from bci.domain import TrialRecord
from bci.splitting.base import SplitPolicy


class SessionHoldoutSplit(SplitPolicy):
    def assign(self, trials: Sequence[TrialRecord]) -> dict[str, str]:
        sessions = sorted({t.session for t in trials})
        if len(sessions) < 2:
            raise ValueError("session holdout requires at least two sessions")
        train_session = sessions[0]
        test_session = sessions[-1]
        return {
            t.trial_id: ("calibration" if t.session == train_session else "test" if t.session == test_session else "validation")
            for t in trials
        }
