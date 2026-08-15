from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from bci.config import BCIConfig
from bci.domain import BCIEvent, EEGChunk, EEGMetadata, RecordingRef
from bci.buffering.trials import map_native_label, normalize_label
from bci.sources.base import DatasetAdapter, EEGSource, EventSource, StreamPublisher


class MOABBReplayPublisher(StreamPublisher):
    def __init__(self, config: BCIConfig, adapter: DatasetAdapter, ref: RecordingRef | None = None):
        self.config = config
        self.adapter = adapter
        self.ref = ref
        self._player = None

    def start(self) -> None:
        from mne_lsl.player import PlayerLSL

        ref = self.ref or next(self.adapter.iter_recordings())
        raw = self.adapter.load_raw(ref)
        replay = self.config.source.replay
        self._player = PlayerLSL(
            raw,
            chunk_size=replay.chunk_size_samples,
            name=replay.stream_name,
            source_id=replay.source_id,
            annotations=replay.annotations,
            annotations_encoding=replay.annotations_encoding,
        )
        self._player.start()

    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()
            self._player = None


class RawReplayEEGSource(EEGSource):
    def __init__(self, config: BCIConfig, raw):
        self.config = config
        self.raw = raw
        self.sfreq = float(raw.info["sfreq"])
        self.ch_names = _selected_eeg_channels(config, raw)
        self.cursor = 0
        self._connected = False

    def connect(self) -> EEGMetadata:
        self.cursor = 0
        self._connected = True
        return EEGMetadata(self.sfreq, list(self.ch_names), "raw-replay")

    def read_latest(self, seconds: float) -> EEGChunk:
        n = max(1, int(round(seconds * self.sfreq)))
        return self._read(n)

    def poll_new(self) -> EEGChunk | None:
        if not self._connected:
            raise RuntimeError("RawReplayEEGSource is not connected")
        if self.cursor >= self.raw.n_times:
            return None
        return self._read(self.config.source.replay.chunk_size_samples)

    def _read(self, n_samples: int) -> EEGChunk | None:
        start = self.cursor
        stop = min(start + n_samples, self.raw.n_times)
        if stop <= start:
            return None
        self.cursor = stop
        data = self.raw.get_data(picks=self.ch_names, start=start, stop=stop)
        times = np.arange(start, stop, dtype=float) / self.sfreq
        return EEGChunk(data=data, sfreq=self.sfreq, ch_names=list(self.ch_names), t_start=float(times[0]), times=times)

    def iter_events(self) -> Iterable[BCIEvent]:
        return []

    def close(self) -> None:
        self._connected = False


class RawReplayEventSource(EventSource):
    def __init__(self, config: BCIConfig, raw, ref: RecordingRef | None = None):
        self.config = config
        self.raw = raw
        self.ref = ref
        self.current_time = 0.0
        self.index = 0

    def connect(self) -> None:
        self.current_time = 0.0
        self.index = 0

    def advance_to(self, timestamp: float) -> None:
        self.current_time = timestamp

    def poll(self) -> list[BCIEvent]:
        events: list[BCIEvent] = []
        annotations = self.raw.annotations
        while self.index < len(annotations) and float(annotations[self.index]["onset"]) <= self.current_time:
            ann = annotations[self.index]
            native = normalize_label(ann["description"])
            command = map_native_label(self.config, native)
            events.append(
                BCIEvent(
                    timestamp=float(ann["onset"]),
                    duration=float(ann["duration"]),
                    native_label=native,
                    command=command,
                    event_index=self.index,
                    dataset=self.ref.dataset if self.ref else self.config.dataset.name,
                    subject=self.ref.subject if self.ref else (self.config.dataset.subjects[0] if self.config.dataset.subjects else None),
                    session=self.ref.session if self.ref else None,
                    run=self.ref.run if self.ref else None,
                )
            )
            self.index += 1
        return events

    def close(self) -> None:
        pass


def _selected_eeg_channels(config: BCIConfig, raw) -> list[str]:
    if config.channels.include:
        return list(config.channels.include)
    types = raw.get_channel_types()
    ch_names = [ch for ch, kind in zip(raw.ch_names, types) if kind == "eeg"]
    return ch_names or list(raw.ch_names)
