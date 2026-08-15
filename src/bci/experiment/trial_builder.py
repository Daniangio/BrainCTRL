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
    window_index: int = 0


class RealtimeTrialBuilder:
    def __init__(self, config: BCIConfig, split_by_event: dict[int, str] | None = None):
        self.config = config
        self.split_by_event = split_by_event or {}
        self.pending: list[PendingTrial] = []
        self.completed_count = 0

    def add_event(self, event: BCIEvent) -> PendingTrial | None:
        if event.command is None:
            return None
        first_pending: PendingTrial | None = None
        event_index = event.event_index if event.event_index is not None else self.completed_count
        starts = [event.timestamp + self.config.trials.onset_offset_seconds]
        if self.config.experiment.mode == "controller_smoke" and event.duration > self.config.trials.window_seconds:
            starts = []
            start = event.timestamp + self.config.trials.onset_offset_seconds
            latest_end = event.timestamp + event.duration
            while start + self.config.trials.window_seconds <= latest_end + 1e-9:
                starts.append(start)
                start += self.config.trials.inference_stride_seconds
        for window_index, start in enumerate(starts):
            end = start + self.config.trials.window_seconds
            split_key = event_index * 1000 + window_index
            split = self.split_by_event.get(split_key, self.split_by_event.get(event_index, "inference"))
            pending = PendingTrial(event=event, start=start, end=end, split=split, window_index=window_index)
            self.pending.append(pending)
            first_pending = first_pending or pending
        return first_pending

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
            original_idx = event.event_index if event.event_index is not None else self.completed_count
            idx = original_idx * 1000 + pending.window_index
            trial_id = (
                f"{event.dataset or self.config.dataset.name}-"
                f"s{event.subject or (self.config.dataset.subjects[0] if self.config.dataset.subjects else 0)}-"
                f"{event.session or 'stream'}-{event.run or 'stream'}-e{original_idx}-w{pending.window_index}"
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
