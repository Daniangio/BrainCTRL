# BrainCTRL V1 — Realtime Experiment, Interactive Launcher, and GUI

## Purpose

V1 turns the current BrainCTRL prototype into a **genuine realtime experiment system**.

The key goal is not merely to display offline results in a GUI. The goal is to make the complete path

```text
EEG source
  -> LSL
  -> event/trial reconstruction
  -> preprocessing
  -> spectral features
  -> calibration
  -> probabilistic decoder
  -> decision policy
  -> GUI / console / UDP
```

run incrementally in wall-clock time.

The implementation must preserve the central architectural requirement of BrainCTRL:

> The downstream experiment code must not care whether EEG comes from a replayed MOABB recording or from a future physical EEG device.

The GUI and interactive shell launcher are clients of this experiment architecture. They must not contain scientific logic.

---

# 1. Current state and main gap

The existing repository already has good abstractions for:

- dataset adapters;
- LSL publishing and receiving;
- preprocessing;
- spectral feature extraction;
- Bayesian latent decoding;
- calibration batches;
- train/validation/test splitting;
- decision policies;
- console/UDP sinks;
- offline artifact generation.

However, the current `run` path is not yet a realtime experiment.

At present it:

1. starts a MOABB `PlayerLSL`;
2. connects to the EEG LSL stream;
3. reads one short chunk as a probe;
4. shuts down the stream;
5. runs the existing offline evaluation pipeline.

Furthermore, `LSLEEGSource.iter_events()` currently returns no events.

Therefore the first V1 milestone is a proper streaming experiment engine.

---

# 2. V1 user experience

After setup, a user should normally need only:

```bash
./brainctrl.sh
```

The launcher should present an interactive menu.

Example:

```text
BrainCTRL
=========

Dataset:
  1) Kalunga2016
  2) Lee2019 SSVEP
  3) Nakanishi2015

Select dataset [1]:

Subject [1]:

Mode:
  1) Realtime replay
  2) Fast offline evaluation
  3) Bootstrap dataset only
  4) Connect to live LSL source

Select mode [1]:

Decoder:
  1) Bayesian latent
  2) Spectral score
  3) CCA

Select decoder [1]:

Open realtime GUI? [Y/n]:

Experiment summary
------------------
Dataset: Kalunga2016
Subject: 1
Source: MOABB replay -> LSL
Decoder: Bayesian latent
Window: 1.5 s
Calibration batch: 6 trials
GUI: enabled

Start? [Y/n]
```

The shell script should then invoke the Python CLI with explicit options.

The shell script is only a convenience layer. All behavior must remain available directly through Python CLI commands for testing and automation.

---

# 3. Recommended repository additions

```text
.
├── brainctrl.sh
├── docs/
│   ├── EXPERIMENT_GUIDE.md
│   ├── COMPONENTS.md
│   └── REALTIME_ARCHITECTURE.md
└── src/bci/
    ├── experiment/
    │   ├── __init__.py
    │   ├── engine.py
    │   ├── events.py
    │   ├── bus.py
    │   ├── trial_builder.py
    │   └── factory.py
    │
    ├── gui/
    │   ├── __init__.py
    │   ├── app.py
    │   ├── main_window.py
    │   ├── controller.py
    │   └── panels/
    │       ├── signal.py
    │       ├── spectrum.py
    │       ├── probabilities.py
    │       ├── latent.py
    │       ├── calibration.py
    │       └── status.py
    │
    └── sources/
        ├── lsl.py
        └── events.py
```

Do not put experiment state logic inside Qt widgets.

---

# 4. Event-driven experiment architecture

Introduce an internal event bus.

This should be a small in-process publish/subscribe mechanism, not an external message broker.

Example interface:

```python
class EventBus:
    def subscribe(self, event_type, callback) -> None:
        ...

    def publish(self, event) -> None:
        ...
```

The experiment engine publishes immutable event objects.

Suggested event types:

```python
@dataclass(frozen=True)
class StreamConnected:
    metadata: EEGMetadata

@dataclass(frozen=True)
class PhaseChanged:
    old_phase: CalibrationPhase
    new_phase: CalibrationPhase

@dataclass(frozen=True)
class EEGWindowReady:
    chunk: EEGChunk

@dataclass(frozen=True)
class TrialStarted:
    event: BCIEvent

@dataclass(frozen=True)
class TrialCompleted:
    trial: TrialRecord

@dataclass(frozen=True)
class FeatureComputed:
    feature: FeatureRecord

@dataclass(frozen=True)
class CalibrationBatchReady:
    n_batch: int
    n_total: int

@dataclass(frozen=True)
class ModelUpdated:
    model_version: int
    metrics: dict

@dataclass(frozen=True)
class PredictionProduced:
    prediction: Prediction

@dataclass(frozen=True)
class DecisionEmitted:
    decision: Decision

@dataclass(frozen=True)
class ExperimentFinished:
    artifact_dir: str
```

Why this matters:

- the GUI subscribes to events;
- console logging subscribes to events;
- artifact logging subscribes to events;
- tests can inspect emitted events;
- future Godot communication can subscribe to decisions;
- scientific components remain unaware of the UI.

---

# 5. Fix LSL event acquisition first

`PlayerLSL` publishes `Raw.annotations` as a second LSL stream with type `annotations`.

V1 must consume both:

```text
EEG stream
  name = configured replay stream name

annotation stream
  type = annotations
  source_id = same source id
```

Create an event receiver abstraction:

```python
class EventSource(ABC):
    def connect(self) -> None:
        ...

    def poll(self) -> list[BCIEvent]:
        ...

    def close(self) -> None:
        ...
```

Implementation:

```python
class LSLAnnotationSource(EventSource):
    ...
```

Prefer `annotations_encoding="one-hot"` initially because it remains numerical and can be handled through `StreamLSL`.

The event receiver must:

1. identify annotation channels;
2. decode the active channel into the native label;
3. use the LSL sample timestamp as event onset;
4. map native label to BrainCTRL command;
5. preserve raw/native label for provenance.

For replay experiments, configure:

```yaml
source:
  replay:
    chunk_size_samples: 1
```

because accurate alignment between replayed EEG and annotation timestamps is more important than replay throughput in this mode.

Later, throughput can be revisited after alignment tests exist.

---

# 6. Realtime trial reconstruction

Introduce `RealtimeTrialBuilder`.

It receives:

- timestamped EEG chunks;
- timestamped `BCIEvent`s.

When an event occurs at time `t_event`, the desired classification window is:

```text
t_start = t_event + onset_offset_seconds
t_end   = t_start + window_seconds
```

The trial builder must wait until EEG samples through `t_end` are available.

Then it extracts the exact interval from the ring buffer and emits a `TrialRecord`.

Conceptually:

```text
event arrives
     |
     v
pending trial
     |
EEG continues arriving
     |
buffer reaches required end timestamp
     |
     v
TrialRecord
```

Do not sleep for `window_seconds` inside the event handler.

Maintain pending trials and resolve them as data arrive.

The ring buffer therefore needs sample timestamps, not only sample values.

---

# 7. Experiment phases

V1 should use the existing `CalibrationPhase` enum meaningfully.

Recommended state progression:

```text
BOOTSTRAP
    |
    v
CALIBRATING
    |
    v
VALIDATING
    |
    v
FROZEN_TEST
```

For later free-running control:

```text
INFERENCE
```

## Calibration

Trials assigned to calibration are accumulated.

After every configured calibration batch:

```text
new labeled trials
      |
      v
feature extraction
      |
      v
batch buffer
      |
enough examples per class?
      |
      +--- no ---> continue collecting
      |
      yes
      |
      v
decoder.update(...)
      |
      v
ModelUpdated event
```

If `refit_on_all_accumulated_data=true`, every update uses all calibration trials received so far.

## Validation

The decoder is not fitted on validation data.

Validation may be used to determine decision thresholds only if that behavior is explicitly implemented and recorded.

## Frozen test

No model parameters or thresholds may be updated.

This is crucial: a visually impressive realtime demo must not destroy the scientific meaning of the held-out test.

---

# 8. Realtime experiment engine

Add:

```python
class RealtimeExperimentEngine:
    def __init__(
        self,
        config,
        eeg_source,
        event_source,
        preprocessor,
        feature_extractor,
        decoder,
        decision_policy,
        sinks,
        event_bus,
    ):
        ...

    def run(self) -> ExperimentResult:
        ...

    def stop(self) -> None:
        ...
```

Approximate main loop:

```python
while not stopped:
    eeg = eeg_source.poll_new()
    events = event_source.poll()

    ring_buffer.append(eeg)

    for event in events:
        trial_builder.add_event(event)

    completed_trials = trial_builder.resolve(ring_buffer)

    for trial in completed_trials:
        bus.publish(TrialCompleted(trial))

        processed = preprocessor.transform(trial)
        features = feature_extractor.transform(processed)
        bus.publish(FeatureComputed(features))

        route_by_experiment_phase(features)
```

The engine owns experiment sequencing.

The GUI does not.

---

# 9. Realtime GUI

## Technology

Use:

- `PySide6` for the desktop application;
- `pyqtgraph` for realtime scientific plots.

Do not use Matplotlib for continuously updating EEG traces.

Matplotlib can remain for saved offline figures.

The GUI should run in the Qt main thread.

The experiment engine should run in a worker thread or Qt-compatible worker object.

Communication from the engine to the GUI must use thread-safe queued signals/events.

---

# 10. GUI layout

A useful first GUI is a scientific dashboard, not a polished game interface.

Recommended layout:

```text
+-------------------------------------------------------------+
| BrainCTRL | source | subject | phase | model v3 | elapsed   |
+----------------------------+--------------------------------+
| EEG traces                 | Spectrum / SSVEP evidence       |
| rolling last 3-5 seconds   | target peaks + harmonics        |
|                            | 13 Hz / 21 Hz                    |
+----------------------------+--------------------------------+
| Posterior probabilities    | Latent-space view               |
| LEFT  ███████ 0.78         | calibration points              |
| RIGHT ██      0.17         | current point                   |
| NONE  █       0.05         | class centers                   |
+----------------------------+--------------------------------+
| Calibration timeline / experiment status                    |
| trial 18/60 | batch 3 | model v3 | next phase: validation    |
+-------------------------------------------------------------+
| last command: LEFT | confidence .91 | latency 1.62 s         |
+-------------------------------------------------------------+
```

---

# 11. GUI panels

## EEG panel

Show a configurable subset of channels over the latest 3-5 seconds.

Purpose:

- verify acquisition;
- see artifacts;
- see whether the stream is alive.

Do not autoscale each frame independently; that makes visual comparison impossible.

## Spectrum panel

This is the most important panel for V0.

For the current analysis window show:

- PSD or log-power spectrum;
- markers at each stimulus frequency;
- markers at included harmonics;
- optionally the local spectral background used by `local_log_snr`.

The GUI should make the feature extraction intuitive.

If LEFT corresponds to 13 Hz and RIGHT to 21 Hz, the user should literally see why a trial favors one class.

## Probability panel

Plot current:

```text
P(LEFT)
P(RIGHT)
P(NONE)
```

Distinguish:

- instantaneous model posterior;
- smoothed evidence from `DecisionPolicy`;
- emitted command.

These are not the same thing.

## Latent panel

For a two-dimensional decoder show calibration features after projection.

Display:

- calibration samples colored by label;
- latent class means;
- current projected trial;
- optionally covariance ellipses.

If latent dimension is one, show distributions on a horizontal axis rather than inventing a second dimension.

The decoder should expose a public transform method or diagnostic state rather than the GUI reaching into private attributes.

## Calibration panel

Show:

- phase;
- trial count per class;
- current batch size;
- model version;
- validation accuracy when available;
- Fisher/Mahalanobis class separation.

This panel should help answer:

> Is calibration actually learning increasingly separable representations?

---

# 12. Diagnostic API

Do not let the GUI access decoder private members such as `_classes`, `W_`, or `latent_means_` directly.

Introduce a diagnostic structure:

```python
@dataclass(frozen=True)
class DecoderDiagnostics:
    model_version: int
    classes: list[str]
    latent_dim: int
    latent_points: np.ndarray | None
    latent_labels: list[str] | None
    class_centers: dict[str, np.ndarray]
    class_covariances: dict[str, np.ndarray]
    separation: dict[str, float]
```

Add to the decoder interface something like:

```python
def diagnostics(self) -> DecoderDiagnostics:
    ...
```

and, when meaningful:

```python
def transform_latent(self, X: np.ndarray) -> np.ndarray:
    ...
```

This keeps visualization modular across future decoder types.

---

# 13. Interactive shell launcher

Add executable:

```text
brainctrl.sh
```

Requirements:

1. locate repository root;
2. create environment by invoking `setup.sh` if `.venv` does not exist;
3. activate `.venv`;
4. discover available YAML configs under `configs/`;
5. let user select configuration;
6. optionally override:
   - subject;
   - mode;
   - decoder;
   - GUI on/off;
7. print resolved experiment summary;
8. ask for final confirmation;
9. execute Python CLI.

Avoid editing the YAML file in-place.

Pass overrides to the Python CLI:

```bash
python -m bci.cli experiment \
    --config configs/kalunga_v0.yaml \
    --subject 1 \
    --model bayesian_latent \
    --gui
```

The Python CLI should apply overrides to the loaded config object in memory.

The Bash script must contain no dataset/model logic.

---

# 14. CLI changes

Recommended CLI commands:

```text
bci bootstrap
bci evaluate
bci replay
bci inspect
bci experiment
bci gui
```

`experiment`:

- runs the genuine realtime engine;
- optionally starts replay publisher if source mode is `moabb_replay`;
- can run headless.

`gui`:

- equivalent to experiment with GUI enabled;
- primarily convenience.

Useful options:

```text
--config
--subject
--session
--run
--model
--gui / --no-gui
--replay-speed
--max-trials
```

Do not duplicate configuration validation in argparse.

---

# 15. Configuration additions

Example:

```yaml
experiment:
  mode: labeled_replay
  gui: true
  max_trials: null
  phase_progression:
    calibration_fraction: 0.50
    validation_fraction: 0.25
    test_fraction: 0.25

gui:
  enabled: true
  refresh_hz: 15
  eeg_history_seconds: 5.0
  spectrum_max_hz: 50.0
  max_channels_displayed: 8
  show_latent: true
  show_raw_eeg: true

source:
  mode: moabb_replay
  replay:
    chunk_size_samples: 1
```

Do not make GUI configuration mandatory for headless operation.

---

# 16. Offline/realtime equivalence test

This is a required scientific test.

For the same original trials, compare:

```text
offline pipeline
vs
LSL replay pipeline
```

The extracted windows and features should agree within a stated numerical tolerance.

Test:

```python
def test_replay_features_match_offline_features():
    ...
```

This is more valuable than merely testing that an LSL connection can be opened.

If results differ, investigate:

- annotation timestamps;
- onset offsets;
- causal filtering state;
- ring-buffer slicing;
- sampling boundaries.

---

# 17. Required V1 tests

At minimum:

### Unit

- event annotation decoding;
- timestamped ring-buffer slicing;
- pending trial resolution;
- experiment state transitions;
- batch model update;
- diagnostics API;
- config overrides;
- event bus delivery.

### Integration

- synthetic EEG + synthetic events through realtime engine;
- MOABB PlayerLSL -> EEG + annotation receiver;
- offline/replay feature-equivalence test;
- headless `experiment` CLI smoke test.

GUI tests can initially be minimal.

Do not make scientific correctness depend on GUI tests.

---

# 18. Documentation restructure

The current root documentation is too architecture-centric for a new user.

Use:

```text
README.md
docs/
├── EXPERIMENT_GUIDE.md
├── COMPONENTS.md
├── REALTIME_ARCHITECTURE.md
└── DEVELOPING_NEW_COMPONENTS.md
```

## README

Keep short.

Explain:

1. what BrainCTRL is;
2. the core idea;
3. setup;
4. `./brainctrl.sh`;
5. a diagram;
6. links to detailed docs.

## EXPERIMENT_GUIDE.md

Explain the experiment intuitively.

Example narrative:

```text
A subject attends to a 13 Hz visual stimulus.
        ↓
The occipital EEG contains enhanced periodic activity near 13 Hz
and its harmonics.
        ↓
BrainCTRL receives the signal through LSL.
        ↓
A trial window is extracted after the stimulus marker.
        ↓
The spectrum is computed.
        ↓
Relative power around candidate stimulus frequencies becomes
the feature vector.
        ↓
Calibration learns a low-dimensional representation in which
LEFT, RIGHT, and NONE are separated.
        ↓
The Bayesian classifier outputs probabilities.
        ↓
The decision layer accumulates evidence across time.
        ↓
Only sufficiently stable evidence becomes a command.
```

## COMPONENTS.md

For every component explain:

- intuitive purpose;
- inputs;
- outputs;
- current implementation;
- why it is modular;
- what future alternatives may replace it.

## REALTIME_ARCHITECTURE.md

Technical timing, thread model, event bus, LSL streams, state machine.

## DEVELOPING_NEW_COMPONENTS.md

Examples:

- add a new dataset adapter;
- add a new decoder;
- add BrainFlow/OpenBCI;
- add a new feature extractor;
- add a new GUI panel.

---

# 19. Implementation order for Codex

Implement in this exact order.

## Step 1 — event stream

- add annotation LSL receiver;
- set replay chunk size to 1;
- decode replay markers;
- test timestamps.

Acceptance criterion:

> A command-line process prints native event labels and timestamps while EEG is being replayed.

## Step 2 — realtime trial builder

- timestamped ring buffer;
- pending trials;
- exact window extraction.

Acceptance criterion:

> Realtime replay produces the same trial windows as offline extraction.

## Step 3 — realtime experiment engine

- experiment state machine;
- incremental features;
- calibration batches;
- validation/test;
- prediction and decision events.

Acceptance criterion:

> `bci experiment --no-gui` performs a complete replayed experiment without invoking the offline evaluation runner.

## Step 4 — event bus and diagnostics

- typed experiment events;
- decoder diagnostics;
- artifact observer.

Acceptance criterion:

> A console observer can display the whole experiment state without engine-specific hooks.

## Step 5 — GUI

- PySide6;
- pyqtgraph;
- EEG;
- spectrum;
- probabilities;
- calibration status;
- latent space.

Acceptance criterion:

> GUI updates during realtime replay without blocking acquisition.

## Step 6 — interactive launcher

- `brainctrl.sh`;
- config selection;
- optional overrides;
- setup fallback.

Acceptance criterion:

> A fresh user who already cloned the repository can run `./brainctrl.sh` and launch an experiment.

## Step 7 — documentation

- concise README;
- experiment guide;
- component guide;
- architecture guide.

Acceptance criterion:

> A technically literate reader unfamiliar with EEG can explain what information flows through each stage and why LEFT/RIGHT discrimination is possible.

---

# 20. Non-goals for V1

Do not yet implement:

- Godot integration beyond preserving the existing command sink boundary;
- deep neural EEG models;
- online self-supervised learning;
- arbitrary thought decoding;
- hardware-specific OpenBCI logic;
- complex distributed services;
- database servers;
- browser-based GUI.

V1 is successful when replay behaves like a real acquisition experiment and its state is understandable live.

---

# 21. Scientific acceptance criteria

A V1 experiment should save enough information to answer:

1. What exact raw recording and trials were used?
2. Which trials calibrated the model?
3. At which trial did each model update occur?
4. What spectrum/features were produced?
5. How separated were the latent classes after each update?
6. What probability was assigned to each class?
7. What command did the evidence accumulator emit?
8. What was performance on held-out data?
9. Did the realtime replay reproduce offline features?

If the GUI looks good but these questions cannot be answered, V1 is not complete.
