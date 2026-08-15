from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import signal

from bci.config import BCIConfig
from bci.domain import TrialRecord
from bci.preprocessing.base import Preprocessor


class StandardPreprocessor(Preprocessor):
    def __init__(self, config: BCIConfig):
        self.config = config

    def transform(self, trial: TrialRecord) -> TrialRecord:
        data = np.asarray(trial.data, dtype=float).copy()
        if self.config.preprocessing.detrend != "none":
            data = signal.detrend(data, axis=1, type=self.config.preprocessing.detrend)
        bp = self.config.preprocessing.bandpass_hz
        if bp is not None:
            low, high = bp
            nyq = trial.sfreq / 2.0
            high = min(high, nyq * 0.98)
            if low > 0 and high > low:
                sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
                data = signal.sosfilt(sos, data, axis=1)
        notch = self.config.preprocessing.notch_hz
        if notch is not None and notch < trial.sfreq / 2.0:
            b, a = signal.iirnotch(notch, Q=30.0, fs=trial.sfreq)
            data = signal.lfilter(b, a, data, axis=1)
        return replace(trial, data=data)
