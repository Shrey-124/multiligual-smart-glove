"""Feature extraction - mirrors what the ESP32 / phone will compute on a window."""
import numpy as np


def window_features(w):
    """(WINDOW, 11) -> 1-D feature vector (mean, std, min, max, range per channel
    + dominant-frequency of the gyro/accel channels, which separates wave vs shake)."""
    mean, std = w.mean(0), w.std(0)
    mn, mx = w.min(0), w.max(0)
    spec = np.abs(np.fft.rfft(w[:, 5:] - w[:, 5:].mean(0), axis=0))
    dom = spec[1:].argmax(0) + 1                      # dominant frequency bin (Hz at 1 s window)
    energy = spec[1:].max(0)
    return np.concatenate([mean, std, mn, mx, mx - mn, dom, energy])


def featurize(X):
    return np.array([window_features(w) for w in X])
