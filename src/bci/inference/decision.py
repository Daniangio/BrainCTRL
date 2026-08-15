from __future__ import annotations

from abc import ABC, abstractmethod

from bci.config import BCIConfig
from bci.domain import Decision, Prediction


class DecisionPolicy(ABC):
    @abstractmethod
    def update(self, prediction: Prediction) -> Decision: ...


class ExponentialEvidencePolicy(DecisionPolicy):
    def __init__(self, config: BCIConfig):
        self.config = config
        self.q: dict[str, float] | None = None
        self.consecutive = 0
        self.last_command_time = -1e12

    def update(self, prediction: Prediction) -> Decision:
        alpha = self.config.decision.alpha
        if self.q is None:
            self.q = dict(prediction.probabilities)
        else:
            for cls, prob in prediction.probabilities.items():
                self.q[cls] = alpha * prob + (1.0 - alpha) * self.q.get(cls, 0.0)
        command, confidence = max(self.q.items(), key=lambda item: item[1])
        emit = command
        if confidence < self.config.decision.posterior_threshold:
            emit = "NONE"
            self.consecutive = 0
        else:
            self.consecutive += 1
        if self.consecutive < self.config.decision.consecutive_windows:
            emit = "NONE"
        if prediction.timestamp - self.last_command_time < self.config.decision.refractory_seconds:
            emit = "NONE"
        if emit != "NONE":
            self.last_command_time = prediction.timestamp
        return Decision(
            timestamp=prediction.timestamp,
            command=emit if self.config.decision.emit_none or emit != "NONE" else "",
            probabilities=dict(self.q),
            confidence=float(confidence),
            model_version=prediction.model_version,
        )
