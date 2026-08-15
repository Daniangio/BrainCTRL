from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from bci.domain import TrialRecord


class SplitPolicy(ABC):
    @abstractmethod
    def assign(self, trials: Sequence[TrialRecord]) -> dict[str, str]: ...
