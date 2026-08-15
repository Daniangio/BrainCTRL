from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from bci.domain import Prediction


def summarize_predictions(predictions: list[Prediction], classes: list[str], rest_label: str = "NONE") -> dict:
    if not predictions:
        return {"n": 0}
    y_true = [p.true_label for p in predictions]
    y_pred = [p.predicted_label for p in predictions]
    proba = np.asarray([[p.probabilities.get(c, 0.0) for c in classes] for p in predictions])
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    brier = float(np.mean(np.sum((proba - np.eye(len(classes))[ [classes.index(y) for y in y_true] ]) ** 2, axis=1)))
    false_rest = sum(1 for t, p in zip(y_true, y_pred) if t == rest_label and p != rest_label)
    rest_minutes = max(sum(1 for t in y_true if t == rest_label) * 1.5 / 60.0, 1e-9)
    return {
        "n": len(predictions),
        "classes": classes,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "precision": dict(zip(classes, map(float, precision))),
        "recall": dict(zip(classes, map(float, recall))),
        "support": dict(zip(classes, map(int, support))),
        "log_loss": _ordered_log_loss(y_true, proba, classes),
        "brier_score": brier,
        "false_commands_per_minute_rest": float(false_rest / rest_minutes),
    }


def _ordered_log_loss(y_true: list[str], proba: np.ndarray, classes: list[str]) -> float:
    idx = np.asarray([classes.index(y) for y in y_true])
    clipped = np.clip(proba[np.arange(len(y_true)), idx], 1e-15, 1.0)
    return max(0.0, float(-np.mean(np.log(clipped))))
