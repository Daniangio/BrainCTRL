from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from bci.config import BCIConfig, write_resolved_config
from bci.domain import FeatureRecord
from bci.evaluation.metrics import summarize_predictions
from bci.evaluation.runner import build_trials, predict_records, write_environment
from bci.features.alignment import EuclideanAlignment
from bci.features.factory import get_feature_extractor
from bci.models.factory import get_decoder
from bci.preprocessing.standard import StandardPreprocessor
from bci.utils.timing import utc_run_id


def loso_folds(subjects: list[int]) -> list[tuple[int, list[int]]]:
    if len(subjects) < 2:
        raise ValueError("LOSO benchmark requires at least two configured subjects")
    return [(target, [subject for subject in subjects if subject != target]) for target in subjects]


def run_loso_benchmark(config: BCIConfig, run_prefix: str = "loso") -> dict:
    artifact_dir = config.project.artifact_dir / utc_run_id(run_prefix)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, artifact_dir / "config_resolved.yaml")
    write_environment(artifact_dir / "environment.json")

    fold_metrics: dict[str, dict] = {}
    classes = list(config.protocol.classes)
    for target_subject, source_subjects in loso_folds(config.dataset.subjects):
        fold_dir = artifact_dir / f"target_subject_{target_subject}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        variants = {
            "source_only": _run_fold_variant(
                config,
                source_subjects=source_subjects,
                target_subject=target_subject,
                classes=classes,
                fold_dir=fold_dir / "source_only",
                target_unlabeled_ea=False,
            ),
            "target_unlabeled_ea": _run_fold_variant(
                config,
                source_subjects=source_subjects,
                target_subject=target_subject,
                classes=classes,
                fold_dir=fold_dir / "target_unlabeled_ea",
                target_unlabeled_ea=True,
            ),
        }
        (fold_dir / "metrics.json").write_text(json.dumps(variants, indent=2), encoding="utf-8")
        fold_metrics[str(target_subject)] = variants

    summary = {
        "artifact_dir": str(artifact_dir),
        "subjects": config.dataset.subjects,
        "folds": fold_metrics,
        "variants": ["source_only", "target_unlabeled_ea"],
        "macro_balanced_accuracy": {
            "source_only": _mean_variant_metric(fold_metrics, "source_only", "balanced_accuracy"),
            "target_unlabeled_ea": _mean_variant_metric(fold_metrics, "target_unlabeled_ea", "balanced_accuracy"),
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_fold_variant(
    config: BCIConfig,
    source_subjects: list[int],
    target_subject: int,
    classes: list[str],
    fold_dir: Path,
    target_unlabeled_ea: bool,
) -> dict:
    fold_dir.mkdir(parents=True, exist_ok=True)
    variant_config = _variant_config(config, target_unlabeled_ea)
    source_features = _features_for_subjects(variant_config, source_subjects, "calibration")
    target_features = _features_for_subjects(variant_config, [target_subject], "test")
    decoder = get_decoder(variant_config)
    decoder.fit(source_features)
    decoder.save(fold_dir / f"model_v{decoder.model_version:03d}.pkl")
    predictions = predict_records(decoder, target_features)
    metrics = summarize_predictions(predictions, classes)
    metrics["variant"] = "target_unlabeled_ea" if target_unlabeled_ea else "source_only"
    metrics["source_subjects"] = source_subjects
    metrics["target_subject"] = target_subject
    metrics["n_source_features"] = len(source_features)
    metrics["n_target_features"] = len(target_features)
    metrics["target_unlabeled_ea"] = target_unlabeled_ea
    metrics["alignment_enabled"] = variant_config.alignment.enabled
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _variant_config(config: BCIConfig, target_unlabeled_ea: bool) -> BCIConfig:
    variant = config.model_copy(deep=True)
    variant.alignment.enabled = target_unlabeled_ea
    variant.alignment.type = "euclidean" if target_unlabeled_ea else "none"
    return variant


def _features_for_subjects(config: BCIConfig, subjects: list[int], split: str) -> list[FeatureRecord]:
    fold_config = config.model_copy(deep=True)
    fold_config.dataset.subjects = subjects
    trials, _ = build_trials(fold_config)
    preprocessor = StandardPreprocessor(fold_config)
    extractor = get_feature_extractor(fold_config)
    aligner = EuclideanAlignment(fold_config)
    features = [aligner.update_transform(extractor.transform(preprocessor.transform(trial))) for trial in trials]
    return [replace(feature, split=split) for feature in features]


def _mean_variant_metric(fold_metrics: dict[str, dict], variant: str, name: str) -> float | None:
    values = [float(metrics[variant][name]) for metrics in fold_metrics.values() if name in metrics.get(variant, {})]
    if not values:
        return None
    return sum(values) / len(values)
