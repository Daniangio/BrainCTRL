from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from bci.config import load_config
from bci.domain import EEGChunk, EEGMetadata
from bci.experiment.bus import EventBus
from bci.experiment.events import ModelUpdated
from bci.experiment.factory import build_realtime_experiment
from bci.features.spectral import SpectralFeatureExtractor
from bci.models.factory import get_decoder
from bci.preprocessing.streaming import StreamingPreprocessor
from bci.protocol.allocation import allocate_protocol
from tests.unit.test_spectral_features import make_sine


def test_decoder_registry_selects_configured_classes():
    config = load_config("configs/kalunga_v0.yaml")
    config.model.type = "gaussian_latent"
    assert type(get_decoder(config)).__name__ == "GaussianLatentDecoder"
    config.model.type = "bayesian_latent"
    assert type(get_decoder(config)).__name__ == "GaussianLatentDecoder"
    config.model.type = "spectral_score"
    assert type(get_decoder(config)).__name__ == "SpectralScoreDecoder"
    config.model.type = "cca"
    assert type(get_decoder(config)).__name__ == "CCADecoder"


def test_lsl_live_source_mode_does_not_create_moabb_replay():
    config = load_config("configs/kalunga_v0.yaml")
    config.source.mode = "lsl_live"
    config.experiment.mode = "live_lsl"
    config.output.console = False
    managed = build_realtime_experiment(config)
    assert managed.publisher is None


def test_harmonics_above_preprocessing_band_are_omitted():
    config = load_config("configs/kalunga_v0.yaml")
    config.preprocessing.bandpass_hz = (6.0, 50.0)
    record = SpectralFeatureExtractor(config).transform(make_sine(21.0, sfreq=256.0))
    assert all("21Hz:h3" not in name for name in record.feature_names)
    assert any(item["reason"] == "above_preprocessing_band" and item["frequency"] == 63.0 for item in record.omitted_harmonics)


def test_streaming_preprocessor_matches_continuous_causal_filter():
    config = load_config("configs/kalunga_v0.yaml")
    config.preprocessing.notch_hz = None
    config.preprocessing.bandpass_hz = (6.0, 30.0)
    sfreq = 128.0
    times = np.arange(512) / sfreq
    data = (np.sin(2 * np.pi * 13.0 * times) + 0.2 * np.sin(2 * np.pi * 3.0 * times))[None, :]
    streamer = StreamingPreprocessor(config)
    streamer.reset(EEGMetadata(sfreq=sfreq, ch_names=["Oz"], source_name="test"))
    chunks = []
    for start in range(0, data.shape[1], 32):
        chunk_times = times[start : start + 32]
        chunk = EEGChunk(data[:, start : start + 32], sfreq, ["Oz"], float(chunk_times[0]), chunk_times)
        chunks.append(streamer.process_chunk(chunk).data)
    chunked = np.concatenate(chunks, axis=1)
    sos = signal.butter(4, [6.0 / (sfreq / 2.0), 30.0 / (sfreq / 2.0)], btype="bandpass", output="sos")
    continuous = signal.sosfilt(sos, data, axis=1)
    assert np.allclose(chunked, continuous, atol=1e-10)


def test_protocol_allocation_balanced_unique_and_deterministic():
    config = load_config("configs/kalunga_v0.yaml")
    config.experiment.mode = "classifier_smoke"
    from bci.experiment.factory import build_synthetic_sources

    _source, _event_source, _split_by_event, entries1 = build_synthetic_sources(config)
    _source, _event_source, _split_by_event, entries2 = build_synthetic_sources(config)
    assert entries1 == entries2
    ids = [entry.event_id for entry in entries1]
    assert len(ids) == len(set(ids))
    roles_by_class = {(entry.role, entry.command) for entry in entries1}
    for role in {"initial_calibration", "challenge", "final_test"}:
        assert {(role, "LEFT"), (role, "RIGHT"), (role, "NONE")} <= roles_by_class


def test_refit_false_rejected_clearly(tmp_path):
    config = load_config("configs/kalunga_v0.yaml")
    config.experiment.mode = "classifier_smoke"
    config.output.console = False
    config.calibration.refit_on_all_accumulated_data = False
    with pytest.raises(NotImplementedError, match="true incremental updates"):
        build_realtime_experiment(config, artifact_dir=tmp_path).run()


def test_refit_resets_decision_state(tmp_path):
    config = load_config("configs/kalunga_v0.yaml")
    config.experiment.mode = "controller_smoke"
    config.output.console = False
    config.experiment.max_idle_seconds = 2.0
    bus = EventBus()
    updates: list[ModelUpdated] = []
    reset_seen: list[bool] = []

    def on_update(event: ModelUpdated) -> None:
        updates.append(event)
        reset_seen.append(managed.engine.decision_policy.q is None)

    managed = build_realtime_experiment(config, bus=bus, artifact_dir=tmp_path)
    bus.subscribe(ModelUpdated, on_update)
    result = managed.run()
    assert result.model_version >= 1
    assert updates
    assert all(reset_seen)
