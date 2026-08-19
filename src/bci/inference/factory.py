from __future__ import annotations

from bci.config import BCIConfig
from bci.inference.decision import DecisionPolicy, ExponentialEvidencePolicy, MarkovEvidencePolicy


def get_decision_policy(config: BCIConfig) -> DecisionPolicy:
    registry = {
        "exponential_evidence": ExponentialEvidencePolicy,
        "markov_evidence": MarkovEvidencePolicy,
    }
    try:
        return registry[config.decision.type](config)
    except KeyError as exc:
        raise ValueError(f"unsupported decision policy {config.decision.type!r}; available: {sorted(registry)}") from exc
