# Adaptive Multilingual Smart Glove - Sign-to-Speech Simulation
Project Group 17 · AY 2026-27 · Guide: Prof. Apurva Ulhe

Hardware-free simulation of the glove from the Phase-1 presentation:
**5 flex sensors + MPU6050 → feature extraction → ML classifier → confidence check → multilingual speech.**

Gestures: `hello`, `help`, `need_water`, `emergency`, `need_food`
Output languages: English, Hindi, Marathi. `emergency` also triggers a simulated SOS (GPS + contacts).

## IMPORTANT - what the data is
No glove exists yet, so training data is **synthetic**: `glove_sim/gestures.py` generates 11-channel
(5 flex + 3 accel + 3 gyro) 1-second windows at 50 Hz with per-wearer variation, noise, drift and
half-formed signs. Reported accuracy is therefore a *simulation* result (leave-users-out CV), not a
hardware result. The pipeline accepts real recordings unchanged (see below).

## Results (simulated wearers never seen in training)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest | 90.0% | 0.90 |
| SVM (RBF) | 88.8% | 0.89 |

Confidence check at threshold 0.70 (Random Forest): 77.3% correct speech, 21.0% "retry / vibrate",
**only 1.7% wrong speech**; 100% of random non-gesture movement is rejected. See `outputs/`.

## Run
```bash
pip install -r requirements.txt
python scripts/train.py                  # trains, evaluates, saves models/glove_model.joblib
streamlit run app.py                     # interactive simulation for the judges
python scripts/demo_cli.py --lang hi --play   # terminal demo
```
Speech uses gTTS, which needs internet.

## Train on Google Cloud
* **Easiest - Colab:** open `notebooks/train_on_colab.ipynb` in Google Colab (File ▸ Open notebook ▸ GitHub), run all cells.
* **Compute Engine VM:** `bash scripts/vertex_train.sh https://github.com/YOUR_USERNAME/smart-glove-sim.git`
  (needs `gcloud auth login` and a project with billing/free credits; the script deletes the VM afterwards).

## Using real glove data later
Record CSVs named `<user>_<gesture>_<rep>.csv` (e.g. `rohan_need_water_03.csv`) with 11 columns
(flex×5, ax, ay, az, gx, gy, gz), ≥50 rows at 50 Hz, then:
`python scripts/train.py --real data/real`

## Layout
`glove_sim/` simulator, features, model, speech · `scripts/` train + demos · `app.py` Streamlit UI · `notebooks/` Colab
