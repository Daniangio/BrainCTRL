# BrainCTRL — Intuitive Experiment and Component Guide

## What BrainCTRL is trying to do

BrainCTRL is an experimental Brain-Computer Interface framework.

The first target is a visually evoked EEG paradigm such as SSVEP.

Instead of asking a person to physically press LEFT or RIGHT, the experiment associates commands with visual stimuli that oscillate at different frequencies.

For example:

```text
LEFT  -> attend to a 13 Hz stimulus
RIGHT -> attend to a 21 Hz stimulus
NONE  -> rest / no command
```

When the visual system repeatedly receives a periodic stimulus, EEG recorded over visual areas can contain activity related to that periodicity and its harmonics.

BrainCTRL tries to detect this structure from short EEG windows.

---

# 1. The complete experiment in one picture

```text
                  ACQUISITION
                      |
          +-----------+-----------+
          |                       |
   MOABB recording           future EEG headset
   replayed through LSL      streaming through LSL
          |                       |
          +-----------+-----------+
                      |
                      v
                timestamped EEG
                      +
                event markers
                      |
                      v
                 TRIAL BUILDER
                      |
          "take 1.5 s after this cue"
                      |
                      v
                PREPROCESSING
                      |
             remove nuisance signal
                      |
                      v
              SPECTRAL FEATURES
                      |
      evidence near 13 Hz, 21 Hz,
             and their harmonics
                      |
                      v
                  DECODER
                      |
        P(LEFT), P(RIGHT), P(NONE)
                      |
                      v
              DECISION POLICY
                      |
        accumulate stable evidence
                      |
                      v
             LEFT / RIGHT / NONE
                      |
         +------------+------------+
         |            |            |
       GUI          console       Godot
```

The most important architectural rule is:

> Everything after LSL should work the same way for replayed and real EEG.

---

# 2. Why replay a public dataset in realtime?

An offline dataset already contains the EEG signal, so replay may initially appear unnecessary.

It is useful because it lets us develop the exact architecture required by a physical device without owning the hardware yet.

Offline processing often assumes that the entire recording is available at once.

A real BCI never has future EEG.

At time `t`, only samples recorded before `t` exist.

Replaying a recording through Lab Streaming Layer forces BrainCTRL to deal with:

- arrival times;
- buffering;
- event synchronization;
- limited windows;
- incremental calibration;
- realtime predictions.

Later the MOABB replay source can be replaced by an EEG device while preserving the rest of the system.

---

# 3. Dataset adapter

## Intuitive role

The dataset adapter answers:

> "How do I obtain and interpret this particular EEG dataset?"

Different public datasets use different folders, subject IDs, sessions, run names, and event labels.

The rest of BrainCTRL should not know those details.

## Input

A dataset configuration such as:

```yaml
dataset:
  name: Kalunga2016
  subjects: [1]
```

## Output

Standard BrainCTRL recording references and MNE Raw objects.

## Future replacement

A new dataset only needs a new adapter.

It should not require changes to feature extraction or decoding.

---

# 4. Stream publisher

## Intuitive role

The publisher makes stored EEG behave like a live device.

For MOABB replay:

```text
MNE Raw
  |
PlayerLSL
  |
network-like realtime stream
```

It publishes:

- EEG samples;
- event annotations.

## Why LSL?

Lab Streaming Layer gives samples timestamps and provides a common interface for realtime physiological data.

The future hardware path should also terminate in LSL.

---

# 5. EEG source

## Intuitive role

The EEG source is the application's "ear".

It listens to LSL and supplies new EEG samples to the experiment engine.

The experiment engine should not know whether the remote producer is:

- MOABB replay;
- OpenBCI;
- another Python process;
- a vendor application.

---

# 6. Event source

EEG alone does not tell us when the subject was asked to attend LEFT or RIGHT.

The recording therefore also contains markers.

Example:

```text
12.30 s  stimulus 13 Hz begins
16.80 s  rest begins
21.10 s  stimulus 21 Hz begins
```

During replay these annotations are streamed through a second LSL event stream.

The event source converts them into standard `BCIEvent` objects.

---

# 7. Ring buffer

A realtime system continuously receives samples.

Keeping the entire recording forever is unnecessary.

Instead, BrainCTRL stores the recent past in a ring buffer.

Example:

```text
current time: 102.0 s

buffer contains:
92.0 ------------------------------ 102.0
```

If a stimulus started at 100.0 s and we need the interval 100.25-101.75 s, the trial builder can retrieve it from this buffer.

A ring buffer is therefore the bridge between continuous EEG and finite trials.

---

# 8. Trial builder

Suppose a marker appears at:

```text
t = 10.0 s
label = 13 Hz
```

and configuration says:

```yaml
onset_offset_seconds: 0.25
window_seconds: 1.5
```

The desired EEG is:

```text
10.25 s ---------------- 11.75 s
```

The first 250 ms are ignored because the evoked response may need time to develop.

The trial builder waits until samples through 11.75 s have actually arrived.

Only then can it construct the trial.

This prevents future information from leaking into realtime processing.

---

# 9. Preprocessing

Raw EEG contains much more than the SSVEP signal of interest.

Preprocessing can include:

- detrending;
- frequency filtering;
- notch filtering;
- channel selection.

A crucial distinction exists between offline and realtime filters.

An offline zero-phase filter can use samples both before and after a point in time.

A realtime causal filter cannot use future samples.

BrainCTRL should be explicit about this difference.

---

# 10. Spectral feature extraction

This is the core of V0.

For each EEG channel, BrainCTRL computes a Fourier spectrum over a short trial.

If LEFT corresponds to 13 Hz, useful evidence may appear around:

```text
13 Hz
26 Hz
39 Hz
```

These are the fundamental frequency and harmonics.

For RIGHT at 21 Hz:

```text
21 Hz
42 Hz
```

BrainCTRL does not simply use absolute power.

It computes local spectral contrast:

```text
power near target frequency
        minus
power in nearby frequencies
```

Conceptually:

```text
       target
         |
         v
      /\                 strong evidence
_____/  \_____

background ----
```

This reduces sensitivity to global changes in EEG amplitude.

The resulting values across channels and candidate frequencies form a feature vector.

---

# 11. Why a low-dimensional latent space?

The raw feature vector may contain tens of dimensions:

```text
O1:13Hz:h1
O1:26Hz:h2
O1:21Hz:h1
...
Oz:13Hz:h1
...
```

Not every dimension is equally informative.

The decoder therefore learns a projection:

```text
high-dimensional spectrum features
              |
              v
       small latent space
```

The desired geometry is:

```text
       LEFT

    o o o o


                 x x x x
                  RIGHT
```

instead of heavily overlapping classes.

The current Bayesian latent decoder uses Fisher/LDA-like geometry: class means should be separated relative to within-class variability.

This is more meaningful than simply maximizing Euclidean distance between class centers.

---

# 12. Bayesian classification

After projection, BrainCTRL models each command as a probability distribution in latent space.

For a new trial the output is not only:

```text
LEFT
```

but:

```text
LEFT   0.82
RIGHT  0.11
NONE   0.07
```

This is important because EEG is noisy.

A controller should know when the decoder is uncertain.

---

# 13. Calibration

The model is personalized using labeled examples.

BrainCTRL accumulates trials in small batches.

Example:

```text
trials 1-6
    |
first fit

trials 7-12
    |
update using all 12

trials 13-18
    |
update using all 18
```

Each update creates a new model version.

The GUI should expose this process rather than hiding it.

Useful questions are:

- Are class clusters becoming more separated?
- Is validation accuracy increasing?
- Is uncertainty decreasing?
- Are some commands systematically confused?

---

# 14. Validation and test

Calibration, validation, and test have distinct meanings.

## Calibration

Can change model parameters.

## Validation

Can be used to assess choices such as thresholds, but must not be treated as training data unless explicitly designed that way.

## Test

Must remain frozen.

No model update should occur after test trials are revealed.

This remains true even in an interactive realtime demo.

---

# 15. Prediction is not yet a command

EEG predictions fluctuate.

Imagine consecutive windows:

```text
LEFT probability

0.61
0.72
0.58
0.88
0.91
```

Triggering a movement after every maximum-probability prediction would produce erratic control.

The decision policy therefore accumulates evidence over time.

Only when confidence is sufficiently strong and persistent is an action emitted.

So:

```text
decoder prediction
        !=
game command
```

This distinction is essential.

---

# 16. What the GUI should teach us

The GUI is not only visual decoration.

It should make every stage inspectable.

A user should be able to see:

### EEG

Is the stream alive? Are there giant artifacts?

### Spectrum

Is there actually a peak at the stimulus frequency?

### Features

Which frequencies/channels contribute evidence?

### Latent space

Are calibration classes separated?

### Probabilities

How certain is the decoder?

### Decision state

Why was a command emitted or rejected?

### Experiment phase

Are we calibrating, validating, testing, or performing free inference?

This makes the GUI a scientific debugging instrument.

---

# 17. Current V0 experiment

The initial experiment uses Kalunga2016.

A simple mapping is:

```text
13 Hz -> LEFT
21 Hz -> RIGHT
rest  -> NONE
17 Hz -> ignored
```

This is intentionally simple.

Before trying complex machine learning, BrainCTRL should prove that a straightforward spectral representation can recover a signal that is known to exist.

---

# 18. What changes when real hardware arrives?

Very little should change downstream.

Current:

```text
MOABB
  |
MNE Raw
  |
PlayerLSL
  |
BrainCTRL
```

Future:

```text
EEG headset
  |
device acquisition software / BrainFlow
  |
LSL
  |
BrainCTRL
```

The remaining pipeline should stay:

```text
LSL
 -> buffer
 -> trial
 -> preprocessing
 -> features
 -> decoder
 -> decision
 -> output
```

This is why the realtime architecture should be completed before buying hardware.

---

# 19. How a future live calibration experiment differs

A replayed dataset already contains labels and stimulus timings.

With a real participant, BrainCTRL or a stimulus application must generate them.

Example:

```text
show LEFT flicker
    |
emit marker LEFT
    |
record EEG
    |
build labeled trial
```

During calibration the expected command is known.

During actual gameplay, labels disappear and only model predictions remain.

Thus the same experiment engine can support:

```text
CALIBRATING -> labeled cues
INFERENCE   -> unlabeled control
```

---

# 20. Mental model for developers

When changing BrainCTRL, ask which layer the change belongs to.

```text
How do I get this dataset?
    -> DatasetAdapter

How do I put samples on the realtime bus?
    -> StreamPublisher

How do I receive EEG?
    -> EEGSource

How do I receive cue markers?
    -> EventSource

How do I turn continuous EEG into examples?
    -> TrialBuilder

How do I clean EEG?
    -> Preprocessor

How do I represent it numerically?
    -> FeatureExtractor

How do I infer command probabilities?
    -> Decoder

How do I convert noisy probabilities into actions?
    -> DecisionPolicy

Where do commands go?
    -> CommandSink

How is an entire experiment sequenced?
    -> ExperimentEngine

How is state shown?
    -> GUI observer
```

If a class starts answering several of these questions at once, the architecture is becoming too coupled.
