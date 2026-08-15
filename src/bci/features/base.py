from __future__ import annotations

from abc import ABC, abstractmethod

from bci.domain import FeatureRecord, TrialRecord


class FeatureExtractor(ABC):
    @abstractmethod
    def transform(self, trial: TrialRecord) -> FeatureRecord: ...
