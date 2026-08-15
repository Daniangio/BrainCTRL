from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from bci.domain import BCIEvent, EEGChunk, EEGMetadata, RecordingRef


class DatasetAdapter(ABC):
    @abstractmethod
    def ensure_available(self) -> None: ...

    @abstractmethod
    def iter_recordings(self) -> Iterator[RecordingRef]: ...

    @abstractmethod
    def load_raw(self, ref: RecordingRef): ...

    @abstractmethod
    def native_labels(self) -> set[str]: ...

    @abstractmethod
    def metadata(self) -> dict: ...


class StreamPublisher(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class EEGSource(ABC):
    @abstractmethod
    def connect(self) -> EEGMetadata: ...

    @abstractmethod
    def read_latest(self, seconds: float) -> EEGChunk: ...

    @abstractmethod
    def iter_events(self) -> Iterable[BCIEvent]: ...

    @abstractmethod
    def close(self) -> None: ...


class EventSource(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def poll(self) -> list[BCIEvent]: ...

    @abstractmethod
    def close(self) -> None: ...
