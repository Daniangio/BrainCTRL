# BrainCTRL Experiment Guide

BrainCTRL detects whether short EEG windows contain stronger spectral evidence for configured visual stimulus frequencies.

The default V1 experiment uses Kalunga2016:

```text
13 Hz -> LEFT
21 Hz -> RIGHT
rest  -> NONE
17 Hz -> ignored
```

During realtime replay, MOABB data is streamed through LSL as if it came from a device. A separate annotation stream provides stimulus markers. The trial builder waits until the complete configured EEG interval is available, then extracts the window without using future samples.

Pipeline:

```text
event marker
  -> wait for EEG window
  -> causal preprocessing
  -> local spectral log-SNR features
  -> latent Bayesian decoder
  -> posterior probabilities
  -> evidence accumulation
  -> command or NONE
```

Calibration trials can update the decoder. Validation and test trials cannot update parameters. Every run writes features, predictions, calibration history, decisions, metrics, models, and resolved config under `artifacts/`.

## Smoke Tutorials

Two synthetic smoke modes are available before running MOABB:

```bash
python -m bci.cli experiment --config configs/kalunga_v0.yaml --smoke-mode classifier --no-gui
python -m bci.cli experiment --config configs/kalunga_v0.yaml --smoke-mode controller --no-gui
```

`classifier_smoke` uses one prediction per trial and sets `consecutive_windows=1`, so it checks feature extraction and Bayesian decoding directly.

`controller_smoke` uses longer stimulus blocks and sliding windows every configured stride, so it checks posterior smoothing and consecutive-window command emission.

Both modes write a `smoke` section in `metrics.json` explaining the purpose, expected behavior, emitted commands, and decision reasons.
