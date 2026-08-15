from __future__ import annotations

import itertools

import numpy as np

from bci.config import BCIConfig
from bci.domain import BCIEvent, RecordingRef
from bci.buffering.trials import map_native_label, normalize_label
from bci.sources.base import EventSource


class LSLAnnotationSource(EventSource):
    def __init__(self, config: BCIConfig, ref: RecordingRef | None = None):
        self.config = config
        self.ref = ref
        self._stream = None
        self._seen: set[tuple[float, str]] = set()
        self._counter = itertools.count()

    def connect(self) -> None:
        from mne_lsl.stream import StreamLSL

        replay = self.config.source.replay
        self._stream = StreamLSL(
            bufsize=self.config.source.lsl.buffer_seconds,
            stype="annotations",
            source_id=replay.source_id,
        )
        self._stream.connect(acquisition_delay=self.config.source.lsl.acquisition_delay_seconds, processing_flags="all")

    def poll(self) -> list[BCIEvent]:
        if self._stream is None:
            raise RuntimeError("LSLAnnotationSource is not connected")
        n_new = int(getattr(self._stream, "n_new_samples", 0))
        if n_new <= 0:
            return []
        data, times = self._stream.get_data(winsize=n_new)
        labels = list(getattr(self._stream, "ch_names", self._stream.info["ch_names"]))
        return decode_one_hot_annotations(
            self.config,
            data,
            times,
            labels,
            seen=self._seen,
            counter=self._counter,
            ref=self.ref,
        )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.disconnect()
            self._stream = None


class SyntheticEventSource(EventSource):
    def __init__(self, events: list[BCIEvent]):
        self.events = sorted(events, key=lambda e: e.timestamp)
        self.index = 0
        self.current_time = 0.0

    def connect(self) -> None:
        self.index = 0

    def advance_to(self, timestamp: float) -> None:
        self.current_time = timestamp

    def poll(self) -> list[BCIEvent]:
        out: list[BCIEvent] = []
        while self.index < len(self.events) and self.events[self.index].timestamp <= self.current_time:
            out.append(self.events[self.index])
            self.index += 1
        return out

    def close(self) -> None:
        pass


def decode_one_hot_annotations(
    config: BCIConfig,
    data: np.ndarray,
    times: np.ndarray,
    labels: list[str],
    seen: set[tuple[float, str]] | None = None,
    counter=None,
    ref: RecordingRef | None = None,
) -> list[BCIEvent]:
    seen = seen if seen is not None else set()
    counter = counter if counter is not None else itertools.count()
    events: list[BCIEvent] = []
    for sample_idx, timestamp in enumerate(times):
        column = data[:, sample_idx]
        active = np.flatnonzero(np.abs(column) > 0)
        for channel_idx in active:
            native = normalize_label(labels[channel_idx])
            command = map_native_label(config, native)
            key = (float(timestamp), native)
            if key in seen:
                continue
            seen.add(key)
            value = float(column[channel_idx])
            duration = 0.0 if value < 0 else value
            event_index = next(counter)
            events.append(
                BCIEvent(
                    timestamp=float(timestamp),
                    duration=duration,
                    native_label=native,
                    command=command,
                    event_index=event_index,
                    dataset=ref.dataset if ref else config.dataset.name,
                    subject=ref.subject if ref else (config.dataset.subjects[0] if config.dataset.subjects else None),
                    session=ref.session if ref else None,
                    run=ref.run if ref else None,
                )
            )
    return events
