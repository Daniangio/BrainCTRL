from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from bci.config import BCIConfig
from bci.domain import TrialRecord


@dataclass(frozen=True)
class ProtocolEntry:
    event_id: int
    trial_id: str
    native_label: str
    command: str
    subject: int
    session: str
    run: str
    role: str
    append_round: int | None = None


def allocate_protocol(config: BCIConfig, events: Sequence[TrialRecord]) -> list[ProtocolEntry]:
    by_class: dict[str, list[TrialRecord]] = defaultdict(list)
    for event in events:
        if event.command in config.protocol.classes:
            by_class[event.command].append(event)

    rng = np.random.default_rng(config.project.seed)
    allocations: list[ProtocolEntry] = []
    for command in config.protocol.classes:
        class_events = sorted(by_class.get(command, []), key=lambda e: (e.subject, e.session, e.run, e.event_index))
        if config.protocol.ordering == "balanced_random":
            indices = np.arange(len(class_events))
            rng.shuffle(indices)
            class_events = [class_events[int(i)] for i in indices]

        n_initial = min(config.protocol.initial_calibration_per_class, len(class_events))
        remaining = class_events[n_initial:]
        n_final = min(config.protocol.final_test_per_class, len(remaining))
        final_events = remaining[-n_final:] if n_final else []
        middle = remaining[:-n_final] if n_final else remaining
        n_challenge = min(config.protocol.challenge_per_class, len(middle))
        challenge_events = middle[-n_challenge:] if n_challenge else []
        reserve_events = middle[:-n_challenge] if n_challenge else middle
        reserve_events = reserve_events[: config.protocol.reserve_calibration_per_class]

        for role, role_events, append_round in [
            ("initial_calibration", class_events[:n_initial], None),
            ("reserve_calibration", reserve_events, 1),
            ("challenge", challenge_events, None),
            ("final_test", final_events, None),
        ]:
            for event in role_events:
                allocations.append(_entry(event, role, append_round))

    if config.protocol.ordering == "grouped_by_class":
        return allocations
    if config.protocol.ordering == "original_dataset_order":
        return sorted(allocations, key=lambda e: (e.subject, e.session, e.run, e.event_id))
    return _balanced_interleave(allocations, config)


def protocol_split_map(entries: Sequence[ProtocolEntry], include_reserve_as_calibration: bool = False) -> dict[int, str]:
    role_to_split = {
        "initial_calibration": "calibration",
        "reserve_calibration": "calibration" if include_reserve_as_calibration else "reserve",
        "challenge": "validation",
        "final_test": "test",
    }
    return {entry.event_id: role_to_split[entry.role] for entry in entries}


def write_protocol_manifest(entries: Sequence[ProtocolEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event_id", "trial_id", "native_label", "command", "subject", "session", "run", "role", "append_round"],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.__dict__)


def _entry(event: TrialRecord, role: str, append_round: int | None) -> ProtocolEntry:
    return ProtocolEntry(
        event_id=event.source_event_id if event.source_event_id is not None else event.event_index,
        trial_id=event.trial_id,
        native_label=event.native_label,
        command=event.command,
        subject=event.subject,
        session=event.session,
        run=event.run,
        role=role,
        append_round=append_round,
    )


def _balanced_interleave(entries: list[ProtocolEntry], config: BCIConfig) -> list[ProtocolEntry]:
    ordered: list[ProtocolEntry] = []
    for role in ["initial_calibration", "reserve_calibration", "challenge", "final_test"]:
        role_entries = [e for e in entries if e.role == role]
        by_class: dict[str, list[ProtocolEntry]] = defaultdict(list)
        for entry in role_entries:
            by_class[entry.command].append(entry)
        while any(by_class.values()):
            classes = list(config.protocol.classes)
            rng = np.random.default_rng(config.project.seed + len(ordered))
            rng.shuffle(classes)
            for command in classes:
                if by_class[command]:
                    ordered.append(by_class[command].pop(0))
    return ordered
