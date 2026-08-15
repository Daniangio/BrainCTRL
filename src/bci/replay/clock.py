from __future__ import annotations

import threading
import time


class ReplayClock:
    def __init__(self, speed: float = 0.0):
        self.speed = speed
        self._paused = False
        self._step = threading.Event()
        self._lock = threading.Lock()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False
        self._step.set()

    def step(self) -> None:
        self._step.set()

    def consume_step(self) -> bool:
        if self._step.is_set():
            self._step.clear()
            return True
        return False

    def set_speed(self, speed: float) -> None:
        if speed < 0:
            raise ValueError("replay speed must be >= 0")
        with self._lock:
            self.speed = speed

    def wait_for_chunk(self, simulated_duration: float) -> None:
        with self._lock:
            speed = self.speed
        if speed <= 0:
            return
        time.sleep(max(0.0, simulated_duration / speed))
