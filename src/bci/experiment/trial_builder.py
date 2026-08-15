from __future__ import annotations

from dataclasses import dataclass

from bci.config import BCIConfig
from bci.domain import BCIEvent, TrialRecord
from bci.buffering.ring import TimestampedRingBuffer


@dataclass(frozen=True)
class PendingTrial:
    event: BCIEvent
    start: float
    end: float
    split: str


class RealtimeTrialBuilder:
    def __init__(self, config: BCIConfig, split_by_event: dict[int, str] | None = None):
        self.config = config
        self.split_by_event = split_by_event or {}
        self.pending: list[PendingTrial] = []
        self.completed_count = 0

    def add_event(self, event: BCIEvent) -> PendingTrial | None:
        if event.command is None:
            return None
        start = event.timestamp + self.config.trials.onset_offset_seconds
        end = start + self.config.trials.window_seconds
        split = self.split_by_event.get(event.event_index if event.event_index is not None else self.completed_count, "inference")
        pending = PendingTrial(event=event, start=start, end=end, split=split)
        self.pending.append(pending)
        return pending

    def resolve(self, buffer: TimestampedRingBuffer) -> list[TrialRecord]:
        completed: list[TrialRecord] = []
        still_pending: list[PendingTrial] = []
        expected_samples = int(round(self.config.trials.window_seconds * buffer.sfreq))
        for pending in self.pending:
            if not buffer.has_interval(pending.start, pending.end):
                still_pending.append(pending)
                continue
            chunk = buffer.slice(pending.start, pending.end, expected_samples=expected_samples)
            event = pending.event
            idx = event.event_index if event.event_index is not None else self.completed_count
            trial_id = (
                f"{event.dataset or self.config.dataset.name}-"
                f"s{event.subject or (self.config.dataset.subjects[0] if self.config.dataset.subjects else 0)}-"
                f"{event.session or 'stream'}-{event.run or 'stream'}-e{idx}"
            )
            completed.append(
                TrialRecord(
                    trial_id=trial_id,
                    dataset=event.dataset or self.config.dataset.name,
                    subject=event.subject or (self.config.dataset.subjects[0] if self.config.dataset.subjects else 0),
                    session=event.session or "stream",
                    run=event.run or "stream",
                    event_index=idx,
                    native_label=event.native_label,
                    command=event.command,
                    start_time=pending.start,
                    end_time=pending.end,
                    sfreq=buffer.sfreq,
                    ch_names=list(buffer.ch_names),
                    data=chunk.data,
                    split=pending.split,
                )
            )
            self.completed_count += 1
        self.pending = still_pending
        return completed
