from __future__ import annotations

import time

import numpy as np

from bci.domain import FeatureRecord, Prediction
from bci.models.base import Decoder


def prediction_from_feature(decoder: Decoder, feature: FeatureRecord) -> Prediction:
    probs = decoder.predict(feature)
    label, confidence = max(probs.items(), key=lambda item: item[1])
    return Prediction(
        trial_id=feature.trial_id,
        true_label=feature.label,
        probabilities=probs,
        predicted_label=label,
        confidence=float(confidence),
        model_version=decoder.model_version,
        timestamp=time.time(),
    )


def prediction_from_array(decoder: Decoder, x: np.ndarray, classes: list[str]) -> dict[str, float]:
    probs = decoder.predict_proba(x)[0]
    return dict(zip(classes, map(float, probs)))
