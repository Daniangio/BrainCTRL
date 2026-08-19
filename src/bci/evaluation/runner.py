from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

from bci.buffering.trials import trials_from_raw
from bci.calibration.trainer import CalibrationTrainer
from bci.config import BCIConfig, write_resolved_config
from bci.domain import FeatureRecord, Prediction
from bci.evaluation.metrics import summarize_predictions
from bci.evaluation.plots import save_confusion_matrix
from bci.features.factory import get_feature_extractor
from bci.inference.engine import prediction_from_feature
from bci.models.base import Decoder
from bci.models.factory import get_decoder
from bci.preprocessing.standard import StandardPreprocessor
from bci.registry import get_dataset_adapter
from bci.sources.lsl import LSLEEGSource
from bci.sources.replay import MOABBReplayPublisher
from bci.splitting.chronological import ChronologicalTrialSplit, apply_split
from bci.utils.timing import utc_run_id


def bootstrap_dataset(config: BCIConfig, artifact_dir: Path | None = None) -> dict:
    adapter = get_dataset_adapter(config)
    adapter.ensure_available()
    meta = adapter.metadata()
    out = artifact_dir or config.project.artifact_dir / utc_run_id("bootstrap")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_trials(config: BCIConfig):
    adapter = get_dataset_adapter(config)
    adapter.ensure_available()
    trials = []
    for ref in adapter.iter_recordings():
        raw = adapter.load_raw(ref)
        trials.extend(trials_from_raw(config, ref, raw))
    if not trials:
        raise RuntimeError("no usable labeled trials were extracted")
    return trials, adapter.metadata()


def prepare_features(config: BCIConfig, artifact_dir: Path) -> tuple[list[FeatureRecord], dict]:
    trials, metadata = build_trials(config)
    manifest = ChronologicalTrialSplit(config).assign(trials)
    split_trials = apply_split(trials, manifest)
    preprocessor = StandardPreprocessor(config)
    extractor = get_feature_extractor(config)
    features = [extractor.transform(preprocessor.transform(t)) for t in split_trials]
    write_split_manifest(split_trials, artifact_dir / "split_manifest.csv")
    return features, metadata


def run_evaluation(config: BCIConfig, run_prefix: str = "eval") -> dict:
    artifact_dir = config.project.artifact_dir / utc_run_id(run_prefix)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, artifact_dir / "config_resolved.yaml")
    write_environment(artifact_dir / "environment.json")
    features, metadata = prepare_features(config, artifact_dir)
    (artifact_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    calibration = [r for r in features if r.split == "calibration"]
    validation = [r for r in features if r.split == "validation"]
    test = [r for r in features if r.split == "test"]
    decoder = get_decoder(config)
    trainer = CalibrationTrainer(config, decoder)
    trainer.fit_batches(calibration, artifact_dir)
    if decoder.model_version == 0:
        decoder.fit(calibration)
        decoder.save(artifact_dir / f"model_v{decoder.model_version:03d}.pkl")

    val_predictions = predict_records(decoder, validation)
    test_predictions = predict_records(decoder, test)
    write_predictions(val_predictions, artifact_dir / "predictions_validation.csv")
    write_predictions(test_predictions, artifact_dir / "predictions_test.csv")
    write_history(trainer.history, artifact_dir / "calibration_history.csv")
    classes = list(decoder.classes_)
    metrics = {
        "validation": summarize_predictions(val_predictions, classes),
        "test": summarize_predictions(test_predictions, classes),
        "artifact_dir": str(artifact_dir),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if metrics["test"].get("confusion_matrix"):
        save_confusion_matrix(metrics["test"]["confusion_matrix"], classes, artifact_dir / "confusion_matrix.png")
    return metrics


def run_lsl_replay_then_evaluate(config: BCIConfig) -> dict:
    artifact_dir = config.project.artifact_dir / utc_run_id("run_lsl_probe")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    status = {"attempted": config.source.mode == "moabb_replay", "connected": False, "warning": None}
    if config.source.mode == "moabb_replay":
        adapter = get_dataset_adapter(config)
        try:
            adapter.ensure_available()
            publisher = MOABBReplayPublisher(config, adapter)
            publisher.start()
            time.sleep(0.5)
            source = LSLEEGSource(config)
            meta = source.connect()
            source.read_latest(min(0.25, config.trials.window_seconds))
            source.close()
            publisher.stop()
            status["connected"] = True
            status["metadata"] = meta.__dict__
        except Exception as exc:  # LSL support varies across local platforms.
            status["warning"] = f"{type(exc).__name__}: {exc}"
            try:
                publisher.stop()  # type: ignore[name-defined]
            except Exception:
                pass
    (artifact_dir / "lsl_probe.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    metrics = run_evaluation(config, "run")
    metrics["lsl_probe"] = status
    return metrics


def predict_records(decoder: Decoder, records: list[FeatureRecord]) -> list[Prediction]:
    return [prediction_from_feature(decoder, record) for record in records]


def write_split_manifest(trials, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial_id", "dataset", "subject", "session", "run", "event_index", "native_label", "command", "split"],
        )
        writer.writeheader()
        for t in trials:
            writer.writerow({
                "trial_id": t.trial_id,
                "dataset": t.dataset,
                "subject": t.subject,
                "session": t.session,
                "run": t.run,
                "event_index": t.event_index,
                "native_label": t.native_label,
                "command": t.command,
                "split": t.split,
            })


def write_predictions(predictions: list[Prediction], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["trial_id", "true_label", "predicted_label", "confidence", "model_version", "probabilities"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            writer.writerow({
                "trial_id": p.trial_id,
                "true_label": p.true_label,
                "predicted_label": p.predicted_label,
                "confidence": p.confidence,
                "model_version": p.model_version,
                "probabilities": json.dumps(p.probabilities, sort_keys=True),
            })


def write_history(history: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model_version", "n_records", "record_ids"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_environment(path: Path) -> None:
    payload = {"python": platform.python_version(), "platform": platform.platform()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
