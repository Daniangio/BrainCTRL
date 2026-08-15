from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")


class EventBus:
    def __init__(self):
        self._subscribers: dict[type | object, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: type[T] | object, callback: Callable[[T], None]) -> None:
        self._subscribers[event_type].append(callback)

    def publish(self, event: object) -> None:
        for callback in list(self._subscribers.get(type(event), [])):
            callback(event)
        for callback in list(self._subscribers.get("*", [])):
            callback(event)
