from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bci.domain import OnlineObservation
from bci.models.base import Decoder


class OnlineAdaptor(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, observation: OnlineObservation, decoder: Decoder) -> dict[str, Any]: ...
