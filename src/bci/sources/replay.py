from __future__ import annotations

from bci.config import BCIConfig
from bci.domain import RecordingRef
from bci.sources.base import DatasetAdapter, StreamPublisher


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
