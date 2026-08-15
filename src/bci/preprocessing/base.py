from __future__ import annotations

from abc import ABC, abstractmethod

from bci.domain import TrialRecord


class Preprocessor(ABC):
    @abstractmethod
    def transform(self, trial: TrialRecord) -> TrialRecord: ...
