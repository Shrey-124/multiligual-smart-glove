"""Train + evaluate the gesture classifier.

    python scripts/train.py                 # synthetic data (simulation)
    python scripts/train.py --real data/real # real glove CSVs (when hardware exists)

Evaluation is *leave-users-out* (GroupKFold): the model is always tested on
simulated wearers it never saw in training, which is the honest number.
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glove_sim.gestures import GESTURES, generate_dataset, simulate_noise_window
from glove_sim.features import featurize
from glove_sim.model import CONF_THRESHOLD


def load_real(folder):
    """Expected: one CSV per recording, named <user>_<gesture>_<rep>.csv with 11 columns
    (flex x5, ax, ay, az, gx, gy, gz) at 50 Hz, >= 50 rows. First 50 rows are used."""
    import glob, pandas as pd
    X, y, g = [], [], []
    users = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        user, gest, _ = os.path.basename(f)[:-4].split("_", 2)[0], None, None
        parts = os.path.basename(f)[:-4].split("_")
        user, rest = parts[0], "_".join(parts[1:-1])
        if rest not in GESTURES: continue
        w = pd.read_csv(f).iloc[:50, :11].to_numpy(float)
        if len(w) < 50: continue
        X.append(w); y.append(GESTURES.index(rest)); g.append(users.setdefault(user, len(users)))
    return np.array(X), np.array(y), np.array(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs("outputs", exist_ok=True)

    if a.real:
        Xw, y, groups = load_real(a.real); src = f"real:{a.real}"
    else:
        Xw, y, groups = generate_dataset(a.users, a.reps); src = "synthetic"
    print(f"data: {src}  windows={len(Xw)}  users={len(set(groups))}  classes={GESTURES}")
    X = featurize(Xw)

    candidates = {
        "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(300, random_state=0, n_jobs=-1)),
        "SVM-RBF":      make_pipeline(StandardScaler(), SVC(C=5, probability=True, random_state=0)),
    }
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    results, best, best_f1 = {}, None, -1
    for name, pipe in candidates.items():
        t = time.time()
        pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv)
        proba = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method="predict_proba")
        conf = proba.max(1)
        acc = accuracy_score(y, pred)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        gating = {str(th): dict(spoken_correct=float(((conf >= th) & (pred == y)).mean()),
                                retry_requested=float((conf < th).mean()),
                                spoken_wrong=float(((conf >= th) & (pred != y)).mean()))
                  for th in (0.5, 0.6, 0.7, 0.8)}
        results[name] = dict(accuracy=acc, precision=p, recall=r, f1=f, confidence_gating=gating)
        print(f"{name:13s} acc={acc:.3f} prec={p:.3f} rec={r:.3f} f1={f:.3f}  ({time.time()-t:.1f}s)")
        g7 = gating["0.7"]
        print(f"   @conf>=0.7: correct speech {g7['spoken_correct']:.1%} | retry {g7['retry_requested']:.1%} | wrong speech {g7['spoken_wrong']:.1%}")
        if f > best_f1: best, best_f1, best_pred = name, f, pred

    # confusion matrix of the best model (held-out users)
    cm = confusion_matrix(y, best_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=GESTURES).plot(ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    ax.set_title(f"{best} - held-out users"); fig.tight_layout(); fig.savefig("outputs/confusion_matrix.png", dpi=150)

    # final model on all data
    final = candidates[best].fit(X, y)
    joblib.dump(final, os.path.join(a.out, "glove_model.joblib"))

    # confidence-check test: unknown motions should be rejected (-> vibration)
    rng = np.random.default_rng(123)
    noise = np.array([simulate_noise_window(rng) for _ in range(300)])
    pn = final.predict_proba(featurize(noise)).max(1)
    rejected = float((pn < CONF_THRESHOLD).mean())
    print(f"best={best}  unknown-motion rejected by confidence check: {rejected:.1%} (threshold {CONF_THRESHOLD})")

    summary = dict(data=src, n_windows=int(len(Xw)), best_model=best, results=results,
                   unknown_rejection_rate=rejected, confidence_threshold=CONF_THRESHOLD, gestures=GESTURES)
    json.dump(summary, open("outputs/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
