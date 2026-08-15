from __future__ import annotations

import json
import socket

from bci.config import UDPConfig
from bci.domain import Decision
from bci.sinks.base import CommandSink


class UDPCommandSink(CommandSink):
    def __init__(self, config: UDPConfig):
        self.config = config
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def emit(self, decision: Decision) -> None:
        payload = json.dumps(decision.__dict__).encode("utf-8")
        self.sock.sendto(payload, (self.config.host, self.config.port))
