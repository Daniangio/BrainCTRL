# Architecture Specification — Modular Adaptive SSVEP BCI

## 1. Design goals

The implementation must satisfy four properties simultaneously:

1. **Replay-real equivalence:** replayed MOABB EEG traverses the same LSL receiver and downstream pipeline that future live EEG uses.
2. **Scientific reproducibility:** every prediction is traceable to a source trial, configuration, split, feature version, and model version.
3. **Online calibration:** labeled trials can be accumulated continuously and models can be refit/updated in batches.
4. **Replaceable components:** acquisition, preprocessing, features, decoder, split policy, decision policy, and output transport depend on interfaces rather than each other’s concrete implementations.

Do not over-engineer networking or UI in V0. Modularity is required primarily at scientific/BCI boundaries.

---

## 2. Proposed repository tree

```text
.
├── AGENTS.md
├── ARCHITECTURE.md
├── README.md
├── setup.sh
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── kalunga_v0.yaml
│   ├── lee_session_holdout.yaml
│   └── nakanishi_benchmark.yaml
├── data/                       # gitignored; MOABB downloads/cache
├── artifacts/                  # gitignored; experiment outputs
├── logs/                       # gitignored
├── src/
│   └── bci/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── domain.py
│       ├── registry.py
│       ├── sources/
│       │   ├── base.py
│       │   ├── moabb.py
│       │   ├── replay.py
│       │   ├── lsl.py
│       │   └── synthetic.py
│       ├── buffering/
│       │   ├── ring.py
│       │   ├── trials.py
│       │   └── store.py
│       ├── preprocessing/
│       │   ├── base.py
│       │   └── standard.py
│       ├── features/
│       │   ├── base.py
│       │   └── spectral.py
│       ├── models/
│       │   ├── base.py
│       │   ├── spectral_score.py
│       │   ├── bayesian_latent.py
│       │   └── cca.py
│       ├── calibration/
│       │   ├── state_machine.py
│       │   └── trainer.py
│       ├── splitting/
│       │   ├── base.py
│       │   ├── chronological.py
│       │   └── session_holdout.py
│       ├── inference/
│       │   ├── engine.py
│       │   └── decision.py
│       ├── sinks/
│       │   ├── base.py
│       │   ├── console.py
│       │   └── udp.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── runner.py
│       │   └── plots.py
│       └── utils/
│           ├── logging.py
│           ├── hashing.py
│           └── timing.py
└── tests/
    ├── unit/
    └── integration/
```

Use a normal `src/` layout and package metadata in `pyproject.toml`.

---

## 3. Component contracts

The exact Python syntax can differ, but preserve these boundaries.

### 3.1 DatasetAdapter

Purpose: dataset-specific access only.

```python
class DatasetAdapter(ABC):
    @abstractmethod
    def ensure_available(self) -> None: ...

    @abstractmethod
    def iter_recordings(self) -> Iterator[RecordingRef]: ...

    @abstractmethod
    def load_raw(self, ref: RecordingRef) -> mne.io.BaseRaw: ...

    @abstractmethod
    def native_labels(self) -> set[str]: ...
```

Implement:

- `Kalunga2016Adapter`
- `Lee2019SSVEPAdapter`
- `Nakanishi2015Adapter`

A registry maps config names to adapters.

### 3.2 StreamPublisher

Purpose: put a recording/device onto LSL.

```python
class StreamPublisher(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
```

V0:

- `MOABBReplayPublisher` backed by `PlayerLSL`.

Future:

- `BrainFlowPublisher` or vendor-specific publisher.

### 3.3 EEGSource

Purpose: consume a standardized stream.

```python
class EEGSource(ABC):
    @abstractmethod
    def connect(self) -> EEGMetadata: ...
    @abstractmethod
    def read_latest(self, seconds: float) -> EEGChunk: ...
    @abstractmethod
    def iter_events(self) -> Iterable[BCIEvent]: ...
    @abstractmethod
    def close(self) -> None: ...
```

Primary implementation: `LSLEEGSource`.

A `SyntheticEEGSource` is useful for tests.

Do not expose MNE-LSL objects outside this adapter.

### 3.4 Preprocessor

```python
class Preprocessor(ABC):
    @abstractmethod
    def transform(self, chunk: EEGChunk) -> EEGChunk: ...
```

The same preprocessing implementation must be callable in both streaming and fast offline evaluation.

Be explicit about whether a filter is causal. For any result described as real-time-feasible, do not silently use future samples or zero-phase filtering.

### 3.5 FeatureExtractor

```python
class FeatureExtractor(ABC):
    @abstractmethod
    def transform(self, trial: TrialRecord) -> FeatureRecord: ...
```

`SpectralFeatureExtractor` should expose interpretable feature names, e.g.:

```text
Oz:13Hz:fundamental
Oz:26Hz:h2
Oz:13Hz:local_snr
...
```

### 3.6 Decoder

```python
class Decoder(ABC):
    @abstractmethod
    def fit(self, records: Sequence[FeatureRecord]) -> None: ...
    def update(self, records: Sequence[FeatureRecord]) -> None:
        return self.fit(records)
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    @property
    @abstractmethod
    def classes_(self) -> Sequence[str]: ...
```

Every fit/update increments `model_version`.

### 3.7 SplitPolicy

```python
class SplitPolicy(ABC):
    @abstractmethod
    def assign(self, manifest: Sequence[TrialMeta]) -> SplitManifest: ...
```

The returned manifest is persisted before any model fitting.

### 3.8 DecisionPolicy

Decoder probabilities are not commands.

```python
class DecisionPolicy(ABC):
    @abstractmethod
    def update(self, prediction: Prediction) -> Decision: ...
```

V0 `ExponentialEvidencePolicy` maintains posterior evidence, threshold, dwell time, and refractory period.

### 3.9 CommandSink

```python
class CommandSink(ABC):
    @abstractmethod
    def emit(self, decision: Decision) -> None: ...
```

V0 implementations: console and UDP.

---

## 4. Data and event flow

### 4.1 Replay mode

```text
Kalunga2016Adapter
     |
     | load Raw + annotations
     v
MOABBReplayPublisher (PlayerLSL)
     |
     +---- EEG LSL stream
     +---- annotations LSL stream
                  |
                  v
             LSLEEGSource
                  |
          streaming ring buffer
                  |
       TrialAssembler / event logic
                  |
              TrialStore
          ________|________
         |                 |
 calibration path      inference path
         |                 |
 feature extractor    feature extractor
         |                 |
 batch trainer        decoder posterior
         |                 |
 model registry       decision policy
                           |
                       CommandSink
```

### 4.2 Future hardware mode

Only the producer changes:

```text
Headset -> BrainFlow/vendor -> LSL -> LSLEEGSource -> [identical remainder]
```

If a headset already exposes LSL, no custom publisher is needed.

---

## 5. Dataset bootstrapping

### 5.1 Default: Kalunga2016

Use `moabb.datasets.Kalunga2016`.

Current MOABB exposes four labels: `13`, `17`, `21`, and `rest` with 8 EEG channels at 256 Hz. The first experiment should request only subject 1 by default.

Call a project-level bootstrap service that:

1. sets MOABB download directory from config;
2. instantiates adapter/dataset;
3. validates subject IDs;
4. calls the dataset download/load mechanism for those subjects only;
5. inspects the resulting recordings and annotations;
6. writes `artifacts/bootstrap_<timestamp>/dataset_metadata.json`.

Do not have `setup.sh` download EEG automatically: environment creation and potentially large/remote data acquisition are separate concerns. `setup.sh` installs the software; `bootstrap` obtains the requested dataset.

This still leaves first use as two simple commands and makes the data choice configuration-driven.

### 5.2 Lee2019_SSVEP

Use later for cross-session calibration/drift tests. Current MOABB metadata describes 54 subjects, 2 sessions, 62 EEG channels, 1000 Hz, and four SSVEP frequencies.

Important current API caveat: `Lee2019_SSVEP(test_run=True)` can expose an online/test run whose SSVEP trial labels are not available through MOABB. Do **not** build supervised evaluation around unlabeled runs. For labeled session-holdout evaluation, use labeled offline/train recordings from separate sessions or whichever labeled recordings are actually exposed by the installed MOABB version; assert labels exist before constructing a supervised test split.

### 5.3 Nakanishi2015

Use as a harder multi-frequency benchmark after V0. Keep the same decoder API; configuration selects a subset or all classes.

---

## 6. Trial assembly

A trial is the atomic split unit.

`TrialAssembler` receives annotation/event onset plus the EEG ring buffer and creates a trial only when the entire configured analysis interval is available.

Recommended V0 config:

- start offset after cue/stimulus onset: configurable, e.g. `0.25 s`;
- analysis window: `1.5 s` initially;
- class mapping: configured;
- ignored native classes: configured.

If the analysis interval extends past available labeled data, skip with a logged reason rather than padding silently.

For fast offline evaluation, the same interval extraction logic may operate directly on MNE recordings instead of wall-clock LSL, but it must create the same `TrialRecord` fields.

---

## 7. Spectral feature design

Let the selected channel time series be `x_c[t]` with sampling rate `Fs`.

For each channel:

1. detrend;
2. apply Hann window;
3. calculate one-sided spectrum/periodogram;
4. convert power to log domain with numerical epsilon;
5. for each configured stimulus frequency `f` and harmonic `h`, calculate a narrow-band feature at `h f` if below Nyquist;
6. optionally subtract average log power from neighboring bins excluding the target bin/guard band.

A robust initial scalar evidence for frequency `f` can be:

`S_f = sum_c sum_h w[c,h] * local_log_SNR(c, h*f)`

A trivial binary latent can be:

`z = S_left - S_right`

This is a required sanity-check baseline before learning a projection.

Keep feature extraction independent from the command mapping: features correspond to physical frequencies/channels; config/model maps them to commands.

---

## 8. Bayesian latent decoder

### 8.1 V0 projection

Input: spectral feature vector `x in R^D`.

Learn a regularized linear discriminant projection:

`z = W^T x`

Use shrinkage covariance / regularization because calibration sample counts will be small relative to feature dimensionality.

For two active commands a one-dimensional projection is expected. With an explicit rest/reject class, allow 2 dimensions.

### 8.2 Probabilistic class model

Model each class in latent space and output posterior probabilities.

Acceptable V0 implementations:

- Gaussian class distributions with shared/shrinkage covariance + class priors;
- Bayesian normal model in 1D;
- Normal-Inverse-Wishart / multivariate Student-t posterior predictive if implemented cleanly.

The abstraction should permit later replacement by a richer Bayesian metric model.

Do not use a raw center-distance loss as the sole objective. Between-class separation must be normalized by within-class uncertainty.

### 8.3 Update semantics

`fit` starts from calibration data.

`update` in V0 may simply refit on all calibration records accumulated so far. Persist:

- calibration record IDs;
- model version;
- fitted projection;
- class means/covariances/priors;
- validation metrics;
- timestamp/config hash.

Later versions can implement true posterior sequential updating without changing calling code.

---

## 9. Calibration and evaluation protocol

For each run:

1. construct complete trial manifest;
2. assign whole trials to splits;
3. save split manifest;
4. stream/replay in chronological order;
5. calibration trials enter `TrialStore` and can update the model only after configured batch boundaries;
6. validation trials never update the model;
7. validation chooses posterior/rejection/evidence policy thresholds;
8. freeze model/policy;
9. test trials are processed exactly once and never feed training;
10. persist predictions before computing summary metrics.

For debugging, permit an `offline_fast=true` mode that processes the same trial manifest without waiting real time.

---

## 10. Artifacts per run

Each experiment directory should contain approximately:

```text
artifacts/<run_id>/
├── config_resolved.yaml
├── environment.json
├── dataset_metadata.json
├── split_manifest.csv
├── calibration_history.csv
├── model_v001.*
├── model_v002.*
├── predictions_validation.csv
├── predictions_test.csv
├── metrics.json
├── confusion_matrix.png
├── calibration_curve.png
├── window_sweep.png              # when requested
├── latent_projection.png         # for 1D/2D latent models
└── run.log
```

`environment.json` should include Python/package versions and git commit if available.

---

## 11. Expected experiments

### Experiment A — spectral sanity check

Subject 1, Kalunga2016.

Compare distributions of simple `13-Hz evidence - 21-Hz evidence` between the mapped LEFT and RIGHT trials. Include NONE separately.

Expected result: active classes should show a detectable shift. If not, investigate channel selection, event alignment, preprocessing, and feature definition before adding model complexity.

### Experiment B — model comparison

Same fixed split, compare:

- spectral score;
- CCA;
- Bayesian latent.

Report both classification metrics and probability/rejection metrics.

### Experiment C — calibration budget

Use increasing numbers of calibration trials/class and evaluate on a fixed validation/test set. Do not regenerate the test set at each budget.

### Experiment D — window-length sweep

Try analysis windows such as 2.0, 1.5, 1.0, 0.75, and 0.5 seconds only when each is valid for the recording. Plot performance vs theoretical acquisition latency.

### Experiment E — session shift

On Lee2019_SSVEP or another suitable labeled multi-session dataset:

- calibrate on earlier session;
- test on later labeled session;
- optionally add small later-session calibration batches;
- measure recovery as a function of new labeled trials.

This is the most relevant experiment for future adaptive real EEG.

---

## 12. What not to implement yet

Do not spend V0 effort on:

- EEGNet/transformers;
- elaborate GUI;
- Godot game logic;
- databases/server infrastructure;
- multiprocessing unless measurement shows need;
- arbitrary plugin systems;
- full multi-user support;
- hardware-specific impedance tools.

Prioritize correctness, reproducibility, and source interchangeability.

---

## 13. Live-data extension contract

When hardware is purchased, adding it should require one of these only:

1. configure an existing hardware LSL stream and set `source.mode: lsl_live`; or
2. implement one new `StreamPublisher` that converts the vendor/BrainFlow stream to LSL.

No changes should be required to:

- trial storage;
- preprocessing;
- spectral features;
- decoder APIs;
- calibration trainer;
- decision policy;
- metrics;
- UDP/Godot sink.

A live experiment will additionally need an `ExperimentStimulusController` that presents/controls the actual visual flicker and emits synchronized LSL event markers. Keep that as a future interface, not part of V0 dataset replay.
