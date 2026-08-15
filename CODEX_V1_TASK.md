# Codex Task — Implement BrainCTRL V1 Realtime Experiment

Read the repository's existing `AGENTS.md`, `ARCHITECTURE.md`, config models, domain types, sources, preprocessing, feature extraction, decoder, calibration, inference, sinks, evaluation code, and tests before editing.

Then read:

- `V1_REALTIME_GUI_PLAN.md`
- `EXPERIMENT_GUIDE.md`

Implement V1 incrementally.

## Critical observation

The current `bci run` path is not a realtime scientific experiment. It only probes LSL and then calls the offline evaluation runner. `LSLEEGSource.iter_events()` is unimplemented.

Do not build the GUI on top of this behavior.

## Required implementation sequence

1. Receive the separate LSL annotation stream emitted by `PlayerLSL`.
2. Change replay experiments to use `chunk_size=1` unless a tested timing-safe alternative is implemented.
3. Add timestamp-aware buffering and realtime trial reconstruction.
4. Add a genuine `RealtimeExperimentEngine`.
5. Add typed experiment events and a small in-process event bus.
6. Run calibration incrementally in wall-clock replay.
7. Run validation and frozen test without parameter leakage.
8. Add decoder diagnostic APIs for latent visualization.
9. Add a PySide6 + pyqtgraph realtime GUI as an observer.
10. Add `brainctrl.sh` as an interactive launcher.
11. Restructure documentation for both intuitive and developer-level understanding.
12. Add tests proving offline/replay equivalence.

## Architectural constraints

- GUI must not own scientific state transitions.
- Bash must not implement scientific logic.
- LSL remains the acquisition boundary.
- Replayed and live EEG must share the same downstream engine.
- Do not let GUI code access decoder private attributes.
- Preserve existing abstractions unless there is a concrete reason to improve them.
- Every streamed trial must preserve provenance.
- Original trials remain the atomic split unit.
- Test data must never update the decoder.
- All new behavior must be usable headlessly.
- Keep Godot integration outside V1 except for preserving `CommandSink`.

## Acceptance command

The target fresh-clone UX is:

```bash
./setup.sh
./brainctrl.sh
```

A user selects Kalunga2016 subject 1, realtime replay, Bayesian latent model, and GUI.

The resulting experiment must show live EEG, spectral evidence, calibration progress, posterior probabilities, and decisions while actually processing data received from LSL.

The experiment must save artifacts sufficient to reproduce and audit the run.

## Required scientific integration test

Implement a test that compares feature vectors obtained from the existing offline extraction path with feature vectors obtained after replaying the same trial through LSL and reconstructing it through the realtime buffer.

They should match within an explicit tolerance, after accounting for any deliberately different causal preprocessing.

This test is a V1 release blocker.
