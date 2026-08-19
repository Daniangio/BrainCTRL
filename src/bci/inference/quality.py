from __future__ import annotations

from bci.domain import Prediction, SignalQuality


def quality_adjust_prediction(
    prediction: Prediction,
    quality: SignalQuality | None,
    hard_reject_threshold: float,
) -> tuple[Prediction, str]:
    if quality is None:
        return prediction, "quality_unavailable"
    classes = list(prediction.probabilities)
    if not classes:
        return prediction, "empty_probabilities"

    hard_reject = quality.score < hard_reject_threshold or "hard_reject" in quality.flags
    weight = 0.0 if hard_reject else max(0.0, min(1.0, quality.score))
    uniform = 1.0 / len(classes)
    adjusted = {label: weight * prediction.probabilities[label] + (1.0 - weight) * uniform for label in classes}
    total = sum(adjusted.values())
    if total > 0.0:
        adjusted = {label: value / total for label, value in adjusted.items()}
    label, confidence = max(adjusted.items(), key=lambda item: item[1])
    if hard_reject:
        action = "hard_reject_uniform"
    elif weight < 0.999:
        action = "attenuated_to_uniform"
    else:
        action = "accepted"
    return (
        Prediction(
            trial_id=prediction.trial_id,
            true_label=prediction.true_label,
            probabilities=adjusted,
            predicted_label=label,
            confidence=float(confidence),
            model_version=prediction.model_version,
            timestamp=prediction.timestamp,
        ),
        action,
    )
