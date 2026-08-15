from __future__ import annotations

from collections.abc import Iterable

from bci.config import BCIConfig
from bci.domain import BCIEvent, EEGChunk, EEGMetadata
from bci.sources.base import EEGSource


class LSLEEGSource(EEGSource):
    def __init__(self, config: BCIConfig):
        self.config = config
        self._stream = None

    def connect(self) -> EEGMetadata:
        from mne_lsl.stream import StreamLSL

        replay = self.config.source.replay
        self._stream = StreamLSL(
            bufsize=self.config.source.lsl.buffer_seconds,
            name=replay.stream_name,
            source_id=replay.source_id,
        )
        self._stream.connect(acquisition_delay=self.config.source.lsl.acquisition_delay_seconds)
        info = self._stream.info
        return EEGMetadata(
            sfreq=float(info["sfreq"]),
            ch_names=list(info["ch_names"]),
            source_name=replay.stream_name,
            source_id=replay.source_id,
        )

    def read_latest(self, seconds: float) -> EEGChunk:
        if self._stream is None:
            raise RuntimeError("LSLEEGSource is not connected")
        data, times = self._stream.get_data(winsize=seconds)
        info = self._stream.info
        return EEGChunk(
            data=data,
            sfreq=float(info["sfreq"]),
            ch_names=list(info["ch_names"]),
            t_start=float(times[0]) if len(times) else 0.0,
            times=times,
        )

    def poll_new(self) -> EEGChunk | None:
        if self._stream is None:
            raise RuntimeError("LSLEEGSource is not connected")
        n_new = int(getattr(self._stream, "n_new_samples", 0))
        if n_new <= 0:
            return None
        data, times = self._stream.get_data(winsize=n_new)
        info = self._stream.info
        if data.size == 0 or len(times) == 0:
            return None
        return EEGChunk(
            data=data,
            sfreq=float(info["sfreq"]),
            ch_names=list(info["ch_names"]),
            t_start=float(times[0]),
            times=times,
        )

    def iter_events(self) -> Iterable[BCIEvent]:
        return []

    def close(self) -> None:
        if self._stream is not None:
            self._stream.disconnect()
            self._stream = None
