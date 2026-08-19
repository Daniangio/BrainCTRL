from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from bci.config import BCIConfig
from bci.domain import Decision, Prediction


class DecisionPolicy(ABC):
    @abstractmethod
    def update(self, prediction: Prediction) -> Decision: ...

    def reset(self) -> None:
        pass


class ExponentialEvidencePolicy(DecisionPolicy):
    def __init__(self, config: BCIConfig):
        self.config = config
        self.q: dict[str, float] | None = None
        self.consecutive = 0
        self.last_candidate: str | None = None
        self.last_command_time = -1e12

    def reset(self) -> None:
        self.q = None
        self.consecutive = 0
        self.last_candidate = None
        self.last_command_time = -1e12

    def update(self, prediction: Prediction) -> Decision:
        alpha = self.config.decision.alpha
        if self.q is None:
            self.q = dict(prediction.probabilities)
        else:
            for cls, prob in prediction.probabilities.items():
                self.q[cls] = alpha * prob + (1.0 - alpha) * self.q.get(cls, 0.0)
        command, confidence = max(self.q.items(), key=lambda item: item[1])
        if command != self.last_candidate:
            self.consecutive = 0
            self.last_candidate = command
        emit = command
        reason = "emitted"
        if confidence < self.config.decision.posterior_threshold:
            emit = "NONE"
            self.consecutive = 0
            reason = "below_threshold"
        else:
            self.consecutive += 1
        if reason == "emitted" and self.consecutive < self.config.decision.consecutive_windows:
            emit = "NONE"
            reason = f"waiting_consecutive_{self.consecutive}_of_{self.config.decision.consecutive_windows}"
        if reason == "emitted" and prediction.timestamp - self.last_command_time < self.config.decision.refractory_seconds:
            emit = "NONE"
            reason = "refractory"
        if emit != "NONE":
            self.last_command_time = prediction.timestamp
        return Decision(
            timestamp=prediction.timestamp,
            command=emit if self.config.decision.emit_none or emit != "NONE" else "",
            probabilities=dict(self.q),
            confidence=float(confidence),
            model_version=prediction.model_version,
            reason=reason,
            threshold=self.config.decision.posterior_threshold,
            consecutive=self.consecutive,
            required_consecutive=self.config.decision.consecutive_windows,
        )


class MarkovEvidencePolicy(DecisionPolicy):
    def __init__(self, config: BCIConfig):
        self.config = config
        self.classes = list(config.protocol.classes)
        self.posterior: dict[str, float] | None = None
        self.consecutive = 0
        self.last_candidate: str | None = None
        self.last_command_time = -1e12
        self.last_switch_time = -1e12

    def reset(self) -> None:
        self.posterior = None
        self.consecutive = 0
        self.last_candidate = None
        self.last_command_time = -1e12
        self.last_switch_time = -1e12

    def update(self, prediction: Prediction) -> Decision:
        observation = self._observation(prediction)
        if self.posterior is None:
            prior = {cls: 1.0 / len(self.classes) for cls in self.classes}
        else:
            prior = self._transition(self.posterior)
        posterior_raw = {cls: prior.get(cls, 0.0) * observation.get(cls, 0.0) for cls in self.classes}
        total = sum(posterior_raw.values())
        if total <= 0.0:
            self.posterior = {cls: 1.0 / len(self.classes) for cls in self.classes}
        else:
            self.posterior = {cls: value / total for cls, value in posterior_raw.items()}

        command, confidence = max(self.posterior.items(), key=lambda item: item[1])
        previous_candidate = self.last_candidate
        if command != self.last_candidate:
            self.consecutive = 0
            self.last_candidate = command
            self.last_switch_time = prediction.timestamp
        emit = command
        reason = "markov_emitted"
        if confidence < self.config.decision.posterior_threshold:
            emit = "NONE"
            self.consecutive = 0
            reason = "markov_below_threshold"
        else:
            self.consecutive += 1
        if (
            reason == "markov_emitted"
            and self.config.decision.mode == "held_state"
            and previous_candidate is not None
            and prediction.timestamp - self.last_switch_time < self.config.decision.switch_hold_seconds
        ):
            emit = "NONE"
            reason = "markov_switch_hold"
        if reason == "markov_emitted" and self.consecutive < self.config.decision.consecutive_windows:
            emit = "NONE"
            reason = f"markov_waiting_consecutive_{self.consecutive}_of_{self.config.decision.consecutive_windows}"
        if reason == "markov_emitted" and prediction.timestamp - self.last_command_time < self.config.decision.refractory_seconds:
            emit = "NONE"
            reason = "markov_refractory"
        if emit != "NONE":
            self.last_command_time = prediction.timestamp
        return Decision(
            timestamp=prediction.timestamp,
            command=emit if self.config.decision.emit_none or emit != "NONE" else "",
            probabilities=dict(self.posterior),
            confidence=float(confidence),
            model_version=prediction.model_version,
            reason=reason,
            threshold=self.config.decision.posterior_threshold,
            consecutive=self.consecutive,
            required_consecutive=self.config.decision.consecutive_windows,
        )

    def _observation(self, prediction: Prediction) -> dict[str, float]:
        values = np.asarray([prediction.probabilities.get(cls, 0.0) for cls in self.classes], dtype=float)
        temperature = max(self.config.decision.observation_temperature, 1.0e-6)
        values = np.power(np.clip(values, 1.0e-12, 1.0), 1.0 / temperature)
        total = float(np.sum(values))
        if total <= 0.0:
            values = np.full(len(self.classes), 1.0 / len(self.classes))
        else:
            values /= total
        return dict(zip(self.classes, map(float, values)))

    def _transition(self, posterior: dict[str, float]) -> dict[str, float]:
        transitioned = {cls: 0.0 for cls in self.classes}
        n = len(self.classes)
        for previous, previous_prob in posterior.items():
            self_prob = (
                self.config.decision.self_transition_none
                if previous == self.config.commands.reject_command
                else self.config.decision.self_transition_active
            )
            off_prob = (1.0 - self_prob) / max(n - 1, 1)
            for current in self.classes:
                transitioned[current] += previous_prob * (self_prob if current == previous else off_prob)
        total = sum(transitioned.values())
        if total > 0.0:
            transitioned = {cls: value / total for cls, value in transitioned.items()}
        return transitioned
