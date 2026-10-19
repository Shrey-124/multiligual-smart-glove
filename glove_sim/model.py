import json, joblib, numpy as np
from .gestures import GESTURES
from .features import window_features

CONF_THRESHOLD = 0.70   # below this the glove vibrates / asks to retry instead of speaking


class GloveClassifier:
    def __init__(self, path="models/glove_model.joblib", threshold=CONF_THRESHOLD):
        self.pipe = joblib.load(path)
        if hasattr(self.pipe[-1], 'n_jobs'): self.pipe[-1].n_jobs = 1   # fast single-window inference
        self.threshold = threshold

    def predict_window(self, w):
        """Returns (gesture_name | None, confidence, all_probs). None = low confidence."""
        p = self.pipe.predict_proba(window_features(w)[None])[0]
        i = int(p.argmax())
        return (GESTURES[i] if p[i] >= self.threshold else None), float(p[i]), p
