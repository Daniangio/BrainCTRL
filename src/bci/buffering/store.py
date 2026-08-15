from __future__ import annotations

from dataclasses import dataclass, field

from bci.domain import TrialRecord


@dataclass
class TrialStore:
    records: list[TrialRecord] = field(default_factory=list)

    def add(self, record: TrialRecord) -> None:
        self.records.append(record)

    def by_split(self, split: str) -> list[TrialRecord]:
        return [r for r in self.records if r.split == split]
