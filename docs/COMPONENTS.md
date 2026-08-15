# Components

## Dataset Adapter

Loads MOABB datasets, validates configured subjects, and exposes MNE Raw recordings. Dataset-specific logic remains here.

## Stream Publisher

`MOABBReplayPublisher` publishes a Raw object to LSL using `PlayerLSL`, including annotations. V1 uses `chunk_size_samples: 1` for timing-safe replay.

## EEG Source

`LSLEEGSource` receives timestamped EEG chunks from LSL. The experiment engine does not know whether the producer is MOABB or hardware.

## Event Source

`LSLAnnotationSource` receives the separate one-hot annotation LSL stream and converts active channels into `BCIEvent` objects.

## Timestamped Buffer

`TimestampedRingBuffer` stores recent samples and LSL timestamps, then slices exact trial intervals.

## Trial Builder

`RealtimeTrialBuilder` queues events and emits `TrialRecord`s only after the complete EEG window has arrived.

## Preprocessor and Features

`StreamingPreprocessor` performs stateful causal filtering on continuous chunks before buffering. `StandardPreprocessor` still supports offline/trial transforms. `SpectralFeatureExtractor` computes relative log-power evidence near stimulus frequencies and valid harmonics, omitting harmonics above Nyquist or the effective preprocessing band.

## Decoder

`GaussianLatentDecoder` learns an LDA-like latent projection and Gaussian class conditionals. The legacy `bayesian_latent` model name remains as an alias. Public diagnostics expose latent centers, points, and separation.

## Decision Policy

`ExponentialEvidencePolicy` smooths posterior probabilities and emits commands only after confidence/dwell/refractory rules are satisfied.

## GUI

The PySide6/pyqtgraph GUI subscribes to experiment events. It displays state but does not own calibration, splits, or model updates.
