from __future__ import annotations

import json

from bci.domain import Decision
from bci.sinks.base import CommandSink


class ConsoleCommandSink(CommandSink):
    def emit(self, decision: Decision) -> None:
        print(json.dumps(decision.__dict__, sort_keys=True))
