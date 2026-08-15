from __future__ import annotations

from pathlib import Path

import numpy as np

from bci.buffering.trials import trials_from_raw
from bci.config import BCIConfig
from bci.domain import BCIEvent
from bci.experiment.bus import EventBus
from bci.experiment.engine import RealtimeExperimentEngine, ExperimentResult
from bci.features.spectral import SpectralFeatureExtractor
from bci.inference.decision import ExponentialEvidencePolicy
from bci.models.factory import get_decoder
from bci.preprocessing.standard import StandardPreprocessor
from bci.preprocessing.streaming import StreamingPreprocessor
from bci.protocol.allocation import allocate_protocol, protocol_split_map
from bci.registry import get_dataset_adapter
from bci.sinks.base import CommandSink
from bci.sinks.console import ConsoleCommandSink
from bci.sinks.udp import UDPCommandSink
from bci.sources.events import LSLAnnotationSource, SyntheticEventSource
from bci.sources.lsl import LSLEEGSource
from bci.sources.replay import MOABBReplayPublisher
from bci.sources.synthetic import ScriptedSyntheticEEGSource


class ManagedExperiment:
    def __init__(self, engine: RealtimeExperimentEngine, publisher=None):
        self.engine = engine
        self.publisher = publisher

    def run(self) -> ExperimentResult:
        return self.engine.run()

    def stop(self) -> None:
        self.engine.stop()


def build_realtime_experiment(config: BCIConfig, bus: EventBus | None = None, artifact_dir: Path | None = None) -> ManagedExperiment:
    bus = bus or EventBus()
    sinks: list[CommandSink] = []
    if config.output.console:
        sinks.append(ConsoleCommandSink())
    if config.output.udp.enabled:
        sinks.append(UDPCommandSink(config.output.udp))
    if config.experiment.mode in {"synthetic", "classifier_smoke", "controller_smoke"}:
        if config.experiment.mode in {"synthetic", "classifier_smoke"}:
            config.decision.consecutive_windows = 1
            config.decision.alpha = 1.0
        eeg_source, event_source, split_by_event, protocol_entries = build_synthetic_sources(config)
        publisher = None
    elif config.source.mode == "moabb_replay":
        adapter = get_dataset_adapter(config)
        adapter.ensure_available()
        ref = next(adapter.iter_recordings())
        raw = adapter.load_raw(ref)
        offline_trials = trials_from_raw(config, ref, raw)
        protocol_entries = allocate_protocol(config, offline_trials)
        split_by_event = protocol_split_map(protocol_entries)
        publisher = MOABBReplayPublisher(config, adapter, ref)
        eeg_source = LSLEEGSource(config)
        event_source = LSLAnnotationSource(config, ref)
    elif config.source.mode == "lsl_live":
        if config.experiment.mode != "live_lsl":
            raise ValueError("source.mode=lsl_live requires experiment.mode=live_lsl")
        publisher = None
        eeg_source = LSLEEGSource(config)
        event_source = LSLAnnotationSource(config)
        split_by_event = {}
        protocol_entries = []
    else:
        raise ValueError(f"unsupported source mode {config.source.mode!r}")
    engine = RealtimeExperimentEngine(
        config=config,
        eeg_source=eeg_source,
        event_source=event_source,
        preprocessor=StandardPreprocessor(config, apply_filters=False),
        feature_extractor=SpectralFeatureExtractor(config),
        decoder=get_decoder(config),
        decision_policy=ExponentialEvidencePolicy(config),
        sinks=sinks,
        event_bus=bus,
        split_by_event=split_by_event,
        streaming_preprocessor=StreamingPreprocessor(config),
        protocol_entries=protocol_entries,
        publisher=publisher,
        artifact_dir=artifact_dir,
    )
    return ManagedExperiment(engine, publisher)


def build_synthetic_sources(config: BCIConfig):
    sfreq = 128.0
    ch_names = ["Oz", "O1"]
    mode = "classifier_smoke" if config.experiment.mode == "synthetic" else config.experiment.mode
    repeats = 4 if mode == "classifier_smoke" else 5
    commands = [("13", "LEFT", 13.0), ("21", "RIGHT", 21.0), ("rest", "NONE", 7.0)] * repeats
    base_trials = _synthetic_trial_records_from_commands(config, commands, sfreq, ch_names)
    protocol_entries = allocate_protocol(config, base_trials)
    command_by_event = {trial.event_index: commands[trial.event_index] for trial in base_trials}
    duration = (
        config.trials.window_seconds + config.trials.onset_offset_seconds
        if mode == "classifier_smoke"
        else 4.0
    )
    gap = duration + 0.35
    total_seconds = gap * len(protocol_entries) + 1.0
    times = np.arange(int(total_seconds * sfreq)) / sfreq
    rng = np.random.default_rng(config.project.seed)
    data = _background_signal(times, len(ch_names), rng, config.experiment.synthetic_difficulty)
    events: list[BCIEvent] = []
    for schedule_idx, entry in enumerate(protocol_entries):
        native, command, freq = command_by_event[entry.event_id]
        onset = 0.5 + schedule_idx * gap
        events.append(
            BCIEvent(
                timestamp=onset,
                duration=duration,
                native_label=native,
                command=command,
                event_index=entry.event_id,
                dataset="Synthetic",
                subject=1,
                session="0",
                run="0",
            )
        )
        start = onset + config.trials.onset_offset_seconds
        stop = onset + duration
        mask = (times >= start) & (times < stop)
        _inject_trial_signal(data, times, mask, command, freq, rng, config.experiment.synthetic_difficulty)
    split_manifest = protocol_split_map(protocol_entries)
    split_by_event: dict[int, str] = {}
    for event in events:
        split = split_manifest[event.event_index or 0]
        if mode == "controller_smoke":
            n_windows = _num_controller_windows(config, event.duration)
            for window_index in range(n_windows):
                split_by_event[(event.event_index or 0) * 1000 + window_index] = split
        else:
            split_by_event[event.event_index or 0] = split
    return ScriptedSyntheticEEGSource(data, sfreq, ch_names, chunk_samples=8), SyntheticEventSource(events), split_by_event, protocol_entries


def _background_signal(times: np.ndarray, n_channels: int, rng: np.random.Generator, difficulty: str) -> np.ndarray:
    noise = {"perfect": 0.0, "easy": 0.08, "noisy": 0.22}[difficulty]
    drift = 0.03 * np.sin(2 * np.pi * 1.0 * times)
    data = np.repeat(drift[None, :], n_channels, axis=0)
    if noise:
        data += rng.normal(0.0, noise, size=data.shape)
    return data


def _inject_trial_signal(
    data: np.ndarray,
    times: np.ndarray,
    mask: np.ndarray,
    command: str,
    freq: float,
    rng: np.random.Generator,
    difficulty: str,
) -> None:
    if not np.any(mask):
        return
    if command == "NONE":
        amp = {"perfect": 0.03, "easy": 0.06, "noisy": 0.10}[difficulty]
    else:
        amp = {"perfect": 1.0, "easy": 0.8, "noisy": 0.45}[difficulty]
    harmonic = {"perfect": 0.0, "easy": 0.18, "noisy": 0.12}[difficulty]
    for ch in range(data.shape[0]):
        phase = rng.uniform(0, 2 * np.pi)
        phase2 = rng.uniform(0, 2 * np.pi)
        channel_scale = 1.0 - ch * 0.18
        if command == "NONE":
            data[ch, mask] += amp * np.sin(2 * np.pi * freq * times[mask] + phase)
        else:
            data[ch, mask] += channel_scale * amp * np.sin(2 * np.pi * freq * times[mask] + phase)
            data[ch, mask] += channel_scale * harmonic * np.sin(2 * np.pi * 2 * freq * times[mask] + phase2)


def _num_controller_windows(config: BCIConfig, duration: float) -> int:
    start = config.trials.onset_offset_seconds
    count = 0
    while start + config.trials.window_seconds <= duration + 1e-9:
        count += 1
        start += config.trials.inference_stride_seconds
    return count


def _synthetic_trial_records(config: BCIConfig, events: list[BCIEvent], sfreq: float, ch_names: list[str]):
    from bci.domain import TrialRecord

    n = int(round(config.trials.window_seconds * sfreq))
    return [
        TrialRecord(
            trial_id=f"Synthetic-s1-0-0-e{event.event_index}",
            dataset="Synthetic",
            subject=1,
            session="0",
            run="0",
            event_index=event.event_index or 0,
            native_label=event.native_label,
            command=event.command or "NONE",
            start_time=event.timestamp + config.trials.onset_offset_seconds,
            end_time=event.timestamp + config.trials.onset_offset_seconds + config.trials.window_seconds,
            sfreq=sfreq,
            ch_names=ch_names,
            data=np.zeros((len(ch_names), n)),
            source_event_id=event.event_index,
        )
        for event in events
        if event.command is not None
    ]


def _synthetic_trial_records_from_commands(config: BCIConfig, commands: list[tuple[str, str, float]], sfreq: float, ch_names: list[str]):
    from bci.domain import TrialRecord

    n = int(round(config.trials.window_seconds * sfreq))
    return [
        TrialRecord(
            trial_id=f"Synthetic-s1-0-0-e{idx}",
            dataset="Synthetic",
            subject=1,
            session="0",
            run="0",
            event_index=idx,
            native_label=native,
            command=command,
            start_time=0.0,
            end_time=config.trials.window_seconds,
            sfreq=sfreq,
            ch_names=ch_names,
            data=np.zeros((len(ch_names), n)),
            source_event_id=idx,
        )
        for idx, (native, command, _freq) in enumerate(commands)
    ]
