Yes — and for the general-purpose direction, I would go one step further: **the core GUI should not know anything about 13 Hz, 21 Hz, or even SSVEP.**

The generic spectrum panel should simply visualize the signal representation per channel:

```text
Channel Oz
power
  ^
  |       /\        /\
  |  /\  /  \__/\__/  \__
  +------------------------> frequency

Channel O1
power
  ^
  |    /\             /\
  |___/  \____/\_____/  \__
  +------------------------> frequency

Channel O2
...
```

The axes could be something standard such as PSD in dB/Hz over the configured frequency range. No red lines at stimulus frequencies, no LEFT/RIGHT semantics, no assumption about harmonics.

That would make the panel useful for essentially any EEG task: SSVEP, motor imagery, alpha modulation, sleep-like rhythms, neurofeedback, artifact inspection, or eventually tasks where the classifier uses something completely different.

I would therefore separate **signal diagnostics** from **model interpretation**.

The generic GUI layer would contain something like:

* time series per channel;
* spectrum/PSD per channel;
* model output probabilities;
* latent representation if the model exposes one;
* experiment state.

Then the feature extractor can optionally expose additional diagnostics. For the current SSVEP extractor, that might later be an optional `Feature diagnostics` panel showing local spectral evidence at its selected frequencies. A motor-imagery extractor might instead expose band powers or CSP components. A learned neural model might expose embeddings or attribution scores.

In other words, I would avoid baking this into `SpectrumPanel`:

```python
for freq in config.stimulus_frequencies.values():
    ...
```

which is exactly what the current implementation does.

Instead its contract should be approximately:

```python
SpectrumPanel.update(
    frequencies,
    power_by_channel,
    channel_names,
)
```

and nothing more.

There is one practical issue: if later you have 32–64 channels, plotting 64 separate spectra simultaneously becomes unreadable. So I would design the generic panel now with **small multiples + channel selection**: perhaps 4–8 channels visible at once, selectable from a list, with “all / posterior / custom” being UI convenience rather than scientific assumptions.

And I would change the data object too. Right now `FeatureRecord` stores:

```python
log_power=log_power.mean(axis=0)
```

so the per-channel spectrum is already destroyed before reaching the GUI.

It should instead retain something like:

```python
spectral_power: np.ndarray   # [n_channels, n_freqs]
spectral_freqs: np.ndarray
spectral_channel_names: list[str]
```

The GUI can then decide how many channels to display.

That architecture is cleaner:

```text
EEG
 │
 ├── generic signal inspector
 │     ├── channel traces
 │     └── channel spectra
 │
 └── FeatureExtractor
       │
       ├── model features
       │
       └── optional task-specific diagnostics
```

So yes: **I would remove the SSVEP target-frequency annotations entirely from the default spectrum panel.** They are useful scientifically, but they belong to an optional feature/model interpretation layer, not to the general-purpose signal viewer.
