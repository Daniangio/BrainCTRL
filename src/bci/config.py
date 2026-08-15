from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


Command = Literal["LEFT", "RIGHT", "NONE"]


class ProjectConfig(BaseModel):
    name: str = "ssvep_bci_v0"
    seed: int = 42
    data_dir: Path = Path("./data/moabb")
    artifact_dir: Path = Path("./artifacts")
    log_dir: Path = Path("./logs")


class DatasetConfig(BaseModel):
    name: str
    subjects: list[int] = Field(default_factory=lambda: [1])
    sessions: list[str] | None = None
    runs: list[str] | None = None
    download_if_missing: bool = True


class ReplayConfig(BaseModel):
    stream_name: str = "BCI-EEG-Replay"
    source_id: str = "bci-moabb-replay"
    annotations: bool = True
    annotations_encoding: str = "one-hot"
    repeat: int = 1
    chunk_size_samples: int = 1
    speed: float = 0.0
    allow_pause: bool = True
    allow_step: bool = True


class LSLConfig(BaseModel):
    buffer_seconds: float = 10.0
    acquisition_delay_seconds: float = 0.01


class SourceConfig(BaseModel):
    mode: Literal["moabb_replay", "lsl_live"] = "moabb_replay"
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    lsl: LSLConfig = Field(default_factory=LSLConfig)


class CommandConfig(BaseModel):
    native_to_command: dict[str, Command]
    ignore_native_labels: list[str] = Field(default_factory=list)
    active_commands: list[Command] = Field(default_factory=lambda: ["LEFT", "RIGHT"])
    reject_command: Command = "NONE"

    @model_validator(mode="after")
    def validate_commands(self) -> "CommandConfig":
        if self.reject_command not in self.native_to_command.values():
            raise ValueError("reject_command must appear in native_to_command values")
        if not set(self.active_commands).issubset({"LEFT", "RIGHT"}):
            raise ValueError("active_commands may only contain LEFT/RIGHT in V0")
        return self


class ChannelConfig(BaseModel):
    include: list[str] | None = None


class PreprocessingConfig(BaseModel):
    detrend: Literal["constant", "linear", "none"] = "constant"
    bandpass_hz: tuple[float, float] | None = (6.0, 50.0)
    notch_hz: float | None = 50.0
    causal_for_streaming: bool = True


class TrialConfig(BaseModel):
    onset_offset_seconds: float = 0.25
    window_seconds: float = 1.0
    inference_stride_seconds: float = 0.25


class FeatureConfig(BaseModel):
    type: str = "spectral_relative_power"
    harmonics: list[int] = Field(default_factory=lambda: [1, 2, 3])
    local_band_half_width_hz: float = 0.5
    neighbor_inner_gap_hz: float = 1.0
    neighbor_outer_width_hz: float = 3.0
    log_epsilon: float = 1.0e-12


class SplitConfig(BaseModel):
    type: str = "chronological_trial"
    calibration_fraction: float = 0.5
    validation_fraction: float = 0.25
    test_fraction: float = 0.25
    stratify_if_possible: bool = True
    group_unit: str = "original_trial"

    @model_validator(mode="after")
    def validate_fractions(self) -> "SplitConfig":
        total = self.calibration_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1.0")
        return self


class CalibrationConfig(BaseModel):
    batch_size_trials: int = 6
    minimum_trials_per_class_before_fit: int = 2
    refit_on_all_accumulated_data: bool = True


class ModelConfig(BaseModel):
    type: str = "gaussian_latent"
    latent_dim: int = 2
    projection: str = "shrinkage_lda"
    covariance: str = "shrinkage"
    class_prior: str = "empirical"
    regularization: float = 1.0e-3
    standardize_features: bool = True


class ChallengeConfig(BaseModel):
    metric: str = "balanced_accuracy"
    pass_threshold: float = 0.80
    minimum_events: int = 6
    max_false_commands_per_minute: float = 2.0


class FinalTestConfig(BaseModel):
    locked: bool = True


class ProtocolConfig(BaseModel):
    mode: Literal["scientific", "exploratory"] = "scientific"
    classes: list[Command] = Field(default_factory=lambda: ["LEFT", "RIGHT", "NONE"])
    ordering: Literal["balanced_random", "grouped_by_class", "original_dataset_order"] = "balanced_random"
    initial_calibration_per_class: int = 2
    reserve_calibration_per_class: int = 3
    challenge_per_class: int = 1
    final_test_per_class: int = 1
    append_calibration_per_class: int = 1
    fit_every_new_events: int = 6
    minimum_events_per_class_before_fit: int = 2
    challenge: ChallengeConfig = Field(default_factory=ChallengeConfig)
    final_test: FinalTestConfig = Field(default_factory=FinalTestConfig)


class BaselineConfig(BaseModel):
    spectral_score: bool = True
    cca: bool = True


class DecisionConfig(BaseModel):
    type: str = "exponential_evidence"
    alpha: float = 0.35
    posterior_threshold: float = 0.85
    consecutive_windows: int = 2
    refractory_seconds: float = 0.5
    emit_none: bool = True


class UDPConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 5005


class OutputConfig(BaseModel):
    console: bool = True
    udp: UDPConfig = Field(default_factory=UDPConfig)


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=list)
    save_predictions: bool = True
    save_latent: bool = True


class ExperimentConfig(BaseModel):
    mode: Literal[
        "labeled_replay",
        "offline_fast",
        "bootstrap_only",
        "live_lsl",
        "synthetic",
        "classifier_smoke",
        "controller_smoke",
    ] = "labeled_replay"
    gui: bool = False
    max_trials: int | None = None
    manual_start: bool = False
    poll_interval_seconds: float = 0.01
    max_idle_seconds: float = 15.0
    synthetic_difficulty: Literal["perfect", "easy", "noisy"] = "easy"
    live_preview: bool = True


class GUIConfig(BaseModel):
    enabled: bool = False
    refresh_hz: float = 15.0
    eeg_history_seconds: float = 1.0
    spectrum_max_hz: float = 50.0
    max_channels_displayed: int = 8
    show_latent: bool = True
    show_raw_eeg: bool = True


class BCIConfig(BaseModel):
    project: ProjectConfig
    dataset: DatasetConfig
    source: SourceConfig = Field(default_factory=SourceConfig)
    commands: CommandConfig
    channels: ChannelConfig = Field(default_factory=ChannelConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    trials: TrialConfig = Field(default_factory=TrialConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    baselines: BaselineConfig = Field(default_factory=BaselineConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    gui: GUIConfig = Field(default_factory=GUIConfig)

    @property
    def stimulus_frequencies(self) -> dict[str, float]:
        freqs: dict[str, float] = {}
        ignored = set(self.commands.ignore_native_labels)
        for native, command in self.commands.native_to_command.items():
            if native in ignored or command == self.commands.reject_command:
                continue
            try:
                freqs[command] = float(native)
            except ValueError as exc:
                raise ValueError(f"active native label {native!r} is not numeric") from exc
        return freqs


def load_config(path: str | Path) -> BCIConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BCIConfig.model_validate(data)


def write_resolved_config(config: BCIConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)
