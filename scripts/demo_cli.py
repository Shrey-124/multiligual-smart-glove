"""Terminal demo: 'wear' a simulated glove, perform gestures, hear/see the output.
    python scripts/demo_cli.py --lang hi --play
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glove_sim.gestures import GESTURES, simulate_window, simulate_noise_window, make_user
from glove_sim.model import GloveClassifier
from glove_sim.speech import speak, phrase

ap = argparse.ArgumentParser()
ap.add_argument("--lang", default="en", choices=["en", "hi", "mr"])
ap.add_argument("--play", action="store_true", help="play audio through the speaker")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
clf = GloveClassifier()
user = make_user(rng)                        # a brand-new wearer the model has never seen
script = ["hello", "need_water", "need_food", "help", "NOISE", "emergency"]
print(f"Simulated wearer, output language = {a.lang}\n")
for g in script:
    w = simulate_noise_window(rng) if g == "NOISE" else simulate_window(g, user, rng)
    pred, conf, _ = clf.predict_window(w)
    tag = "random movement" if g == "NOISE" else g
    if pred is None:
        print(f"[{tag:16s}] conf={conf:.2f}  -> LOW CONFIDENCE: glove vibrates, please retry")
    else:
        print(f"[{tag:16s}] conf={conf:.2f}  -> {pred:11s} -> \"{phrase(pred, a.lang)}\"")
        speak(pred, a.lang, play=a.play)
        if pred == "emergency":
            print("                 >> SOS: sending GPS location to registered contacts (simulated)")
