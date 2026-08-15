from __future__ import annotations

from bci.config import BCIConfig
from bci.domain import RecordingRef, TrialRecord


def normalize_label(label: object) -> str:
    text = str(label).strip()
    if text.startswith("Stimulus/"):
        text = text.split("/", 1)[1]
    return text


def map_native_label(config: BCIConfig, native_label: str) -> str | None:
    label = normalize_label(native_label)
    if label in set(config.commands.ignore_native_labels):
        return None
    return config.commands.native_to_command.get(label)


def trials_from_raw(config: BCIConfig, ref: RecordingRef, raw) -> list[TrialRecord]:
    sfreq = float(raw.info["sfreq"])
    if config.channels.include:
        ch_names = list(config.channels.include)
    else:
        types = raw.get_channel_types()
        ch_names = [ch for ch, kind in zip(raw.ch_names, types) if kind == "eeg"]
        if not ch_names:
            ch_names = list(raw.ch_names)
    trials: list[TrialRecord] = []
    for idx, ann in enumerate(raw.annotations):
        native = normalize_label(ann["description"])
        command = map_native_label(config, native)
        if command is None:
            continue
        start_time = float(ann["onset"]) + config.trials.onset_offset_seconds
        end_time = start_time + config.trials.window_seconds
        if ann["duration"] > 0 and end_time > float(ann["onset"]) + float(ann["duration"]):
            continue
        start = max(0, raw.time_as_index(start_time, use_rounding=True)[0])
        stop = raw.time_as_index(end_time, use_rounding=True)[0]
        if stop > raw.n_times or stop <= start:
            continue
        data = raw.get_data(picks=ch_names, start=start, stop=stop)
        if data.shape[1] < int(round(config.trials.window_seconds * sfreq * 0.95)):
            continue
        trial_id = f"{ref.dataset}-s{ref.subject}-{ref.session}-{ref.run}-e{idx}"
        trials.append(
            TrialRecord(
                trial_id=trial_id,
                dataset=ref.dataset,
                subject=ref.subject,
                session=ref.session,
                run=ref.run,
                event_index=idx,
                native_label=native,
                command=command,
                start_time=start_time,
                end_time=end_time,
                sfreq=sfreq,
                ch_names=ch_names,
                data=data,
                source_event_id=idx,
            )
        )
    return trials
