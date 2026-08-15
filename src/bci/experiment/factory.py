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
from bci.models.bayesian_latent import BayesianLatentDecoder
from bci.preprocessing.standard import StandardPreprocessor
from bci.registry import get_dataset_adapter
from bci.sinks.base import CommandSink
from bci.sinks.console import ConsoleCommandSink
from bci.sinks.udp import UDPCommandSink
from bci.sources.events import LSLAnnotationSource, SyntheticEventSource
from bci.sources.lsl import LSLEEGSource
from bci.sources.replay import MOABBReplayPublisher
from bci.sources.synthetic import ScriptedSyntheticEEGSource
from bci.splitting.chronological import ChronologicalTrialSplit, apply_split


class ManagedExperiment:
    def __init__(self, engine: RealtimeExperimentEngine, publisher=None):
        self.engine = engine
        self.publisher = publisher

    def run(self) -> ExperimentResult:
        if self.publisher is not None:
            self.publisher.start()
        try:
            return self.engine.run()
        finally:
            if self.publisher is not None:
                self.publisher.stop()

    def stop(self) -> None:
        self.engine.stop()


def build_realtime_experiment(config: BCIConfig, bus: EventBus | None = None, artifact_dir: Path | None = None) -> ManagedExperiment:
    bus = bus or EventBus()
    sinks: list[CommandSink] = []
    if config.output.console:
        sinks.append(ConsoleCommandSink())
    if config.output.udp.enabled:
        sinks.append(UDPCommandSink(config.output.udp))
    if config.experiment.mode == "synthetic":
        eeg_source, event_source, split_by_event = build_synthetic_sources(config)
        publisher = None
    else:
        adapter = get_dataset_adapter(config)
        adapter.ensure_available()
        ref = next(adapter.iter_recordings())
        raw = adapter.load_raw(ref)
        offline_trials = trials_from_raw(config, ref, raw)
        manifest = ChronologicalTrialSplit(config).assign(offline_trials)
        split_trials = apply_split(offline_trials, manifest)
        split_by_event = {idx: trial.split or "inference" for idx, trial in enumerate(split_trials)}
        publisher = MOABBReplayPublisher(config, adapter, ref)
        eeg_source = LSLEEGSource(config)
        event_source = LSLAnnotationSource(config, ref)
    engine = RealtimeExperimentEngine(
        config=config,
        eeg_source=eeg_source,
        event_source=event_source,
        preprocessor=StandardPreprocessor(config),
        feature_extractor=SpectralFeatureExtractor(config),
        decoder=BayesianLatentDecoder(config),
        decision_policy=ExponentialEvidencePolicy(config),
        sinks=sinks,
        event_bus=bus,
        split_by_event=split_by_event,
        artifact_dir=artifact_dir,
    )
    return ManagedExperiment(engine, publisher)


def build_synthetic_sources(config: BCIConfig):
    sfreq = 128.0
    ch_names = ["Oz", "O1"]
    events: list[BCIEvent] = []
    commands = [("13", "LEFT", 13.0), ("21", "RIGHT", 21.0), ("rest", "NONE", 7.0)] * 4
    gap = config.trials.window_seconds + config.trials.onset_offset_seconds + 0.35
    total_seconds = gap * len(commands) + 1.0
    times = np.arange(int(total_seconds * sfreq)) / sfreq
    data = 0.03 * np.sin(2 * np.pi * 3.0 * times)[None, :]
    data = np.repeat(data, len(ch_names), axis=0)
    for idx, (native, command, freq) in enumerate(commands):
        onset = 0.5 + idx * gap
        events.append(
            BCIEvent(
                timestamp=onset,
                duration=config.trials.window_seconds + config.trials.onset_offset_seconds,
                native_label=native,
                command=command,
                event_index=idx,
                dataset="Synthetic",
                subject=1,
                session="0",
                run="0",
            )
        )
        start = onset + config.trials.onset_offset_seconds
        stop = start + config.trials.window_seconds
        mask = (times >= start) & (times < stop)
        if command != "NONE":
            data[:, mask] += np.sin(2 * np.pi * freq * times[mask])[None, :]
        else:
            data[:, mask] += 0.15 * np.sin(2 * np.pi * freq * times[mask])[None, :]
    split_manifest = ChronologicalTrialSplit(config).assign(_synthetic_trial_records(config, events, sfreq, ch_names))
    split_by_event = {event.event_index or 0: split_manifest[f"Synthetic-s1-0-0-e{event.event_index}"] for event in events}
    return ScriptedSyntheticEEGSource(data, sfreq, ch_names, chunk_samples=8), SyntheticEventSource(events), split_by_event


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
        )
        for event in events
        if event.command is not None
    ]
