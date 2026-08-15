# Realtime Architecture

V1 adds a genuine streaming experiment path:

```text
PlayerLSL / live LSL
  -> LSLEEGSource + LSLAnnotationSource
  -> TimestampedRingBuffer
  -> RealtimeTrialBuilder
  -> RealtimeExperimentEngine
  -> EventBus observers
```

The engine publishes immutable events such as `StreamConnected`, `TrialCompleted`, `FeatureComputed`, `ModelUpdated`, `PredictionProduced`, and `DecisionEmitted`.

Phase progression:

```text
BOOTSTRAP -> CALIBRATING -> VALIDATING -> FROZEN_TEST
```

Calibration records may update the decoder in configured batches. Validation and test records are prediction-only.

The GUI runs the engine in a worker thread. The event bus forwards events through queued Qt signals, so plot updates do not block acquisition.

Artifacts written per experiment include:

- `config_resolved.yaml`
- `protocol_manifest.csv`
- `features.csv`
- `calibration_history.csv`
- `model_vNNN.pkl`
- `predictions_validation.csv`
- `predictions_test.csv`
- `decisions.csv`
- `metrics.json`

Realtime preprocessing is stateful: causal IIR filters process continuous chunks before the ring buffer. Trial-level preprocessing then avoids restarting filters on each window.
