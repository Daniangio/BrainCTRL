# Codex Implementation Instructions — Adaptive SSVEP BCI Prototype

## Mission

Build a small, testable Python project for an adaptive SSVEP brain-computer interface (BCI). The first version must run entirely from public MOABB EEG datasets, but the architecture must be deliberately designed so that replacing replayed data with a live EEG headset does **not** require changes to preprocessing, feature extraction, calibration, decoding, evaluation, or game-control code.

The initial task is binary visual-command decoding plus rejection:

- `LEFT`: attend to one flicker frequency.
- `RIGHT`: attend to another flicker frequency.
- `NONE`: rest / reject / insufficient evidence.

For the default Kalunga2016 experiment map:

- `13 Hz -> LEFT`
- `21 Hz -> RIGHT`
- `rest -> NONE`
- ignore `17 Hz` in V0

Do not assume this mapping in model code. It belongs in configuration.

The project is an experimental research codebase, not a production BCI or medical device.

---

## Core architectural rule

**LSL is the acquisition boundary.**

The offline/replay path must be:

`MOABB -> MNE Raw -> MNE-LSL PlayerLSL -> LSL -> application`

A future live path should be:

`EEG hardware -> vendor/BrainFlow adapter -> LSL -> application`

Everything after LSL must be shared.

Do not build the model around pre-extracted MOABB NumPy arrays. Direct MOABB access is allowed only inside dataset preparation / replay publisher components and offline unit tests.

---

## Bootstrap behavior

The user should be able to set up the repository with one command:

```bash
bash setup.sh
```

`setup.sh` must:

1. fail with a clear message if Python < 3.11;
2. create `.venv` if missing;
3. upgrade pip/setuptools/wheel;
4. install `requirements.txt`;
5. create local directories such as `data/`, `artifacts/`, and `logs/`;
6. run a lightweight import smoke test;
7. print the commands to activate the environment and run the first experiment.

Dataset files must **not** be committed to git.

MOABB should automatically download a configured dataset/subject when first requested. Use `moabb.set_download_dir(...)` or a supported MOABB `path=` mechanism so all downloads go under the configured local data root rather than an uncontrolled home-directory cache.

Provide a CLI bootstrap command such as:

```bash
python -m bci.cli bootstrap --config configs/kalunga_v0.yaml
```

This command should download only the configured subjects by default, validate that the dataset can be loaded, and print basic metadata. Do not download all subjects unless explicitly configured. **However, bootstrap must be optional:** every command that needs data (`evaluate`, `replay`, `run`) must call the same `ensure_available()` logic and automatically download the configured subjects if they are missing. A user must never have to download MOABB data manually.

---

## Implementation order

Implement in this order, with tests after each stage.

### Phase 1 — repository skeleton and config

Create the package structure described in `ARCHITECTURE.md`.

Configuration must be YAML and validated with Pydantic. Avoid scattering experimental constants through code.

At minimum the config controls:

- dataset name;
- data directory;
- subjects;
- sessions/runs if supported;
- source mode (`moabb_replay` initially, `lsl_live` later);
- replay speed and LSL stream names;
- command/frequency mapping;
- selected EEG channels;
- preprocessing;
- spectral feature settings;
- window length and inference stride;
- train/validation/test split policy;
- calibration batch size;
- model type;
- evidence accumulation and rejection threshold;
- random seed;
- artifact/log directories.

### Phase 2 — dataset registry and automatic download

Implement a dataset registry; do not use `if dataset_name == ...` throughout the codebase.

Initial entries:

- `Kalunga2016`
- `Lee2019_SSVEP`
- `Nakanishi2015`

The registry adapter is responsible for:

1. constructing the MOABB dataset class;
2. validating requested subjects/sessions;
3. triggering download to the configured data directory;
4. returning MNE Raw/session/run objects with event annotations preserved;
5. exposing dataset metadata and native class labels.

Do not hard-code trial lengths from documentation. Dataset documentation can contain different notions of task/trial/epoch duration. Use actual annotations/data and the configured analysis window, and fail if the requested epoch exceeds available labeled data.

For Kalunga2016, trust the class labels exposed by the current MOABB version. Current MOABB notes that historical 17/21-Hz event labels were corrected; do not manually swap them again.

### Phase 3 — replay through LSL

Create a `MOABBReplayPublisher` implementing a common publisher interface.

It must:

- load one configured subject/session/run;
- ensure events are represented as MNE annotations;
- create `mne_lsl.player.PlayerLSL` from the `Raw` object;
- stream EEG and annotations;
- use stable configurable LSL `name` and `source_id` values;
- support finite replay for tests and optional repetition for demos;
- support normal real-time speed; accelerated replay is optional and must never be used for latency claims.

Create the shared LSL receiver separately. Use `mne_lsl.stream.StreamLSL` (or lower-level MNE-LSL if required) and a ring buffer.

Do not let downstream components know whether the LSL producer is MOABB or hardware.

### Phase 4 — domain objects and trial buffering

Use typed dataclasses/Pydantic models for objects crossing components. At minimum:

- `EEGMetadata`
- `EEGChunk`
- `BCIEvent`
- `TrialRecord`
- `FeatureRecord`
- `Prediction`
- `CalibrationState`

Each trial must keep provenance:

- dataset;
- subject;
- session;
- run;
- event/trial index;
- original class label/frequency;
- mapped command;
- start/end timestamps;
- split (`calibration`, `validation`, `test`);
- feature extractor version/config hash.

Never split overlapping windows from the same original trial across calibration and test.

### Phase 5 — preprocessing and spectral features

Create abstract interfaces:

- `Preprocessor`
- `FeatureExtractor`

V0 implementation:

1. select configured channels;
2. detrend / remove DC;
3. causal or offline-appropriate bandpass (configured; default approximately 6–50 Hz);
4. optional notch based on data/hardware line frequency;
5. apply Hann window;
6. compute real FFT / periodogram;
7. compute log-power or log relative-power features near stimulus frequencies and harmonics.

The model must not consume arbitrary absolute EEG amplitude without normalization.

Implement a compact interpretable feature representation first. For each candidate stimulus frequency `f`, estimate spectral evidence around `f`, optionally `2f` and `3f` when below Nyquist, normalized against nearby spectral bins.

Do not claim zero-padding increases true spectral resolution.

### Phase 6 — baseline models

Create a `Decoder` base class with at least:

```text
fit(records)
update(records)
predict_proba(features)
predict(features)
save(path)
load(path)
```

Implement these in order:

1. `SpectralScoreDecoder`: deterministic spectral evidence baseline.
2. `BayesianLatentDecoder`: primary V0 model.
3. `CCADecoder`: strong classical SSVEP baseline.

Do not add deep learning until these work and are benchmarked.

#### BayesianLatentDecoder V0

Start simple and interpretable.

Let `x` be the vector of spectral features. Learn a low-dimensional linear projection `z = W x` using a regularized Fisher/LDA-type objective or equivalent shrinkage method. For LEFT/RIGHT, one latent dimension is sufficient; with LEFT/RIGHT/NONE, permit 2D.

Then model class-conditional latent distributions probabilistically. Preferred V0:

- Gaussian class conditionals with shrinkage covariance, or
- a conjugate Bayesian Gaussian model whose posterior predictive is Student-t.

Expose calibrated posterior class probabilities.

Do **not** maximize raw Euclidean distance between class centroids: scale can make that objective arbitrarily large. Optimize separation relative to within-class variance (Fisher/Mahalanobis/Bhattacharyya-like geometry).

`NONE` must be treated explicitly. Also support abstention if posterior confidence is below threshold even if a NONE training class exists.

### Phase 7 — calibration state machine

Implement explicit states:

- `BOOTSTRAP`
- `CALIBRATING`
- `VALIDATING`
- `FROZEN_TEST`
- `INFERENCE`

Calibration behavior:

1. labeled trials arrive sequentially;
2. store them in a persistent batch store;
3. after `N` new calibration trials, update/refit model;
4. evaluate on validation only;
5. select thresholds/hyperparameters using validation only;
6. freeze all model and threshold choices before test;
7. process test trials without parameter updates.

Persist every calibration update so learning curves can be reproduced.

V0 can use batch refitting after each batch. The interface must permit later online Bayesian updates without architectural changes.

### Phase 8 — split policies

Create a `SplitPolicy` base class.

Implement:

- `ChronologicalTrialSplit`: for Kalunga2016 V0; assign whole original trials chronologically to calibration/validation/test.
- `SessionHoldoutSplit`: for datasets such as Lee2019; calibrate on one session and evaluate on a later session when labels permit.
- optionally `BlockHoldoutSplit` if run/block metadata is available.

Random splitting of overlapping windows is prohibited.

The default experiment should be deterministic from `seed` and should write the resolved split manifest to disk before training.

### Phase 9 — real-time inference and evidence accumulation

The online inference service should periodically request the newest configured window from the ring buffer, compute features, and obtain posterior probabilities.

Do not map each instantaneous prediction directly to a game action.

Implement a `DecisionPolicy` base class and an `ExponentialEvidencePolicy`:

`q_t = alpha * p_t + (1-alpha) * q_(t-1)`

Only emit a command when:

- its accumulated probability exceeds a configurable threshold;
- it has remained above threshold for a minimum dwell time or required number of consecutive decisions;
- refractory/debounce rules permit another command.

Otherwise emit `NONE` / abstain.

### Phase 10 — evaluation

At minimum report:

- balanced accuracy;
- confusion matrix;
- per-class precision/recall;
- posterior log loss;
- Brier score or another probability-calibration measure;
- false commands per minute in rest/reject periods;
- command decision latency when meaningful;
- performance vs calibration trials/class;
- performance vs analysis-window length.

Always distinguish offline replay processing speed from actual decision latency.

Save metrics as JSON/CSV plus plots under the run artifact directory.

### Phase 11 — game transport

Do not implement a full game initially.

Create a small `CommandSink` interface and:

- `ConsoleCommandSink` for tests;
- `UDPCommandSink` for Godot.

Example JSON datagram:

```json
{
  "timestamp": 123.456,
  "command": "LEFT",
  "probabilities": {"LEFT": 0.94, "RIGHT": 0.03, "NONE": 0.03},
  "confidence": 0.94,
  "model_version": 4
}
```

Godot integration must remain optional; the BCI pipeline must be testable without Godot.

---

## CLI requirements

Provide one top-level CLI, for example `python -m bci.cli` or installed command `bci`.

Required subcommands:

```bash
# Download/cache configured data and validate metadata
python -m bci.cli bootstrap --config configs/kalunga_v0.yaml

# Run deterministic offline experiment directly from trials/features
python -m bci.cli evaluate --config configs/kalunga_v0.yaml

# Start MOABB as an LSL replay publisher
python -m bci.cli replay --config configs/kalunga_v0.yaml

# Consume LSL, calibrate in chronological pseudo-real-time, then test
python -m bci.cli run --config configs/kalunga_v0.yaml

# Optional: inspect live streams / plot spectrum
python -m bci.cli inspect --config configs/kalunga_v0.yaml
```

The `evaluate` path is allowed to bypass wall-clock LSL replay for fast scientific iteration, but it must use the same preprocessing, feature, split, and decoder implementations as the real-time path.

---

## Testing requirements

Use `pytest`.

At minimum test:

1. config validation;
2. dataset registry lookup;
3. one-subject bootstrap without downloading the whole dataset;
4. command label mapping;
5. trial-level split has no provenance overlap;
6. spectral extractor identifies a synthetic sinusoid at the expected frequency;
7. harmonics above Nyquist are excluded;
8. Bayesian decoder posterior probabilities sum to 1;
9. latent projection cannot improve objective merely by scalar rescaling;
10. abstention works below threshold;
11. batch updates increment model version;
12. serialization/deserialization preserves predictions;
13. LSL replay smoke test when platform supports it;
14. source substitution test: downstream pipeline accepts two fake `EEGSource` implementations unchanged.

Keep unit tests small and independent of full dataset downloads wherever possible. Mark dataset/integration tests separately.

---

## Scientific safeguards

- Do not leak trials across train/validation/test through overlapping windows.
- Do not tune on the test set.
- Record all preprocessing/model configuration with each run.
- Keep subject identity/session/run in provenance.
- Prefer chronological or session-wise generalization tests over random epoch splits.
- Report `NONE` false-positive rate; overall accuracy alone is insufficient for a controller.
- Always benchmark against simple spectral scoring and CCA.
- Do not introduce a neural network merely because the project is described as AI.
- Treat this as BCI research/prototyping, not diagnosis or medical inference.

---

## Definition of V0 done

V0 is done when a fresh checkout can run:

```bash
bash setup.sh
source .venv/bin/activate
python -m bci.cli evaluate --config configs/kalunga_v0.yaml
python -m bci.cli run --config configs/kalunga_v0.yaml
```

with the first data-dependent command automatically downloading/cacheing the configured subject if necessary. `python -m bci.cli bootstrap ...` remains available for explicit pre-download/validation but is not required.

and the final command:

1. replays Kalunga2016 data through LSL;
2. receives it through the same LSL input component intended for live EEG;
3. accumulates complete labeled trials in chronological order;
4. calibrates the Bayesian spectral decoder in batches;
5. freezes the model;
6. evaluates unseen trials;
7. prints and saves posterior-based metrics;
8. optionally emits LEFT/RIGHT/NONE to a console or UDP sink.

The code must make switching to live data a configuration/source-adapter change, not a rewrite.

---

## Current upstream API references

These were checked against current documentation in August 2026. Re-check APIs while implementing rather than assuming signatures forever.

- MOABB install: https://moabb.neurotechx.com/docs/install/install_pip.html
- MOABB `set_download_dir`: https://moabb.neurotechx.com/docs/generated/moabb.set_download_dir.html
- Kalunga2016: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kalunga2016.html
- Lee2019_SSVEP: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_SSVEP.html
- MOABB dataset summary: https://moabb.neurotechx.com/docs/dataset_summary.html
- MNE-LSL `PlayerLSL`: https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.player.PlayerLSL.html
- MNE-LSL `StreamLSL`: https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.stream.StreamLSL.html
- MNE-LSL annotations tutorial: https://mne.tools/mne-lsl/stable/generated/tutorials/20_player_annotations.html
