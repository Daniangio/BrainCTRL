# BrainCTRL

BrainCTRL is an experimental SSVEP brain-computer interface prototype.

It maps visual EEG responses to simple commands:

```text
13 Hz -> LEFT
21 Hz -> RIGHT
rest  -> NONE
```

The central rule is that acquisition ends at Lab Streaming Layer (LSL). Public MOABB recordings are replayed through LSL now; future EEG hardware should also enter through LSL. Everything after that boundary is shared: buffering, trial reconstruction, preprocessing, spectral features, calibration, decoding, decision policy, GUI, and artifacts.

## Quick Start

```bash
./setup.sh
./brainctrl.sh
```

If executable bits are not preserved on your platform, run:

```bash
bash setup.sh
bash brainctrl.sh
```

Useful direct commands:

```bash
python -m bci.cli bootstrap  --config configs/kalunga_v0.yaml
python -m bci.cli evaluate   --config configs/kalunga_v0.yaml
python -m bci.cli experiment --config configs/kalunga_v0.yaml --gui
python -m bci.cli experiment --config configs/kalunga_v0.yaml --synthetic --no-gui
```

## Flow

```text
MOABB Raw -> PlayerLSL -> LSL EEG + annotations
                              |
                              v
timestamped buffer -> trial builder -> preprocessing -> spectral features
                              |
                              v
calibration -> Bayesian latent decoder -> probabilities -> evidence policy
                              |
                              v
GUI / console / UDP artifacts
```

## Documentation

- [Experiment Guide](docs/EXPERIMENT_GUIDE.md)
- [Components](docs/COMPONENTS.md)
- [Realtime Architecture](docs/REALTIME_ARCHITECTURE.md)
- [Developing New Components](docs/DEVELOPING_NEW_COMPONENTS.md)

This is research/prototyping code, not a medical device.
