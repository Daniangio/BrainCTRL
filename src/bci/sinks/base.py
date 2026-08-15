from __future__ import annotations

from abc import ABC, abstractmethod

from bci.domain import Decision


class CommandSink(ABC):
    @abstractmethod
    def emit(self, decision: Decision) -> None: ...
