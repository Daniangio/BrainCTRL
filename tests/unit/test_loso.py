from __future__ import annotations

import pytest

from bci.config import load_config
from bci.evaluation import loso
from bci.evaluation.loso import loso_folds


def test_loso_folds_hold_out_each_subject_once():
    assert loso_folds([1, 2, 3]) == [(1, [2, 3]), (2, [1, 3]), (3, [1, 2])]


def test_loso_folds_require_at_least_two_subjects():
    with pytest.raises(ValueError, match="at least two"):
        loso_folds([1])


def test_loso_benchmark_reports_source_only_and_target_unlabeled_ea(tmp_path, monkeypatch):
    config = load_config("configs/kalunga_v0.yaml")
    config.dataset.subjects = [1, 2]
    config.project.artifact_dir = tmp_path

    def fake_run_fold_variant(config, source_subjects, target_subject, classes, fold_dir, target_unlabeled_ea):
        fold_dir.mkdir(parents=True, exist_ok=True)
        return {
            "balanced_accuracy": 0.75 if target_unlabeled_ea else 0.5,
            "source_subjects": source_subjects,
            "target_subject": target_subject,
            "alignment_enabled": target_unlabeled_ea,
        }

    monkeypatch.setattr(loso, "_run_fold_variant", fake_run_fold_variant)
    summary = loso.run_loso_benchmark(config, run_prefix="loso-test")
    assert summary["variants"] == ["source_only", "target_unlabeled_ea"]
    assert summary["macro_balanced_accuracy"] == {"source_only": 0.5, "target_unlabeled_ea": 0.75}
    assert set(summary["folds"]["1"]) == {"source_only", "target_unlabeled_ea"}
