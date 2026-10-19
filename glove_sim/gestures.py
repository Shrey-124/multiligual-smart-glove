"""Gesture definitions and a synthetic glove-sensor simulator.

The real glove (ESP32) streams 11 channels at ~50 Hz:
    5 flex sensors (thumb..pinky, 0 = straight, 1 = fully bent)
    MPU6050 accelerometer (ax, ay, az, in g)
    MPU6050 gyroscope     (gx, gy, gz, in rad/s)

Until the hardware is ready, `simulate_window` produces windows of that same
shape.  When real data exists, drop CSVs into data/real/ (see README) and the
same training pipeline will use them - nothing else changes.
"""
import numpy as np

GESTURES = ["hello", "help", "need_water", "emergency", "need_food"]

CHANNELS = ["flex_thumb", "flex_index", "flex_middle", "flex_ring", "flex_pinky",
            "ax", "ay", "az", "gx", "gy", "gz"]

FS = 50            # sampling rate (Hz)
WINDOW = 50        # samples per window (1 second)

# Per-gesture template.
#   flex  : mean bend of each finger (0..1)
#   tilt  : static gravity vector (ax, ay, az) in g  -> hand orientation
#   motion: (axis index 0-5 into [ax,ay,az,gx,gy,gz], amplitude, frequency Hz)
GESTURE_SPEC = {
    # open hand, waved side to side
    "hello":      dict(flex=[0.10, 0.05, 0.05, 0.05, 0.08], tilt=[0.0, 0.3, 0.95],
                       motion=[(4, 2.2, 2.5), (1, 0.25, 2.5)]),
    # fist with thumb extended, raised upward in a slow lift
    "help":       dict(flex=[0.15, 0.90, 0.92, 0.90, 0.88], tilt=[0.7, 0.0, 0.7],
                       motion=[(2, 0.35, 1.0), (3, 0.8, 1.0)]),
    # 'W' shape (index/middle/ring up, thumb+pinky bent), small taps at the chin
    "need_water": dict(flex=[0.75, 0.05, 0.08, 0.06, 0.78], tilt=[-0.2, 0.85, 0.45],
                       motion=[(5, 0.6, 3.0), (2, 0.15, 3.0)]),
    # tight fist shaken fast and hard
    "emergency":  dict(flex=[0.92, 0.97, 0.98, 0.97, 0.95], tilt=[0.1, -0.2, 0.97],
                       motion=[(3, 4.5, 6.0), (4, 4.0, 6.0), (0, 0.9, 6.0)]),
    # fingertips pinched together, repeatedly brought to the mouth
    "need_food":  dict(flex=[0.55, 0.50, 0.52, 0.55, 0.58], tilt=[0.35, 0.6, 0.7],
                       motion=[(3, 1.4, 1.5), (1, 0.4, 1.5)]),
}


def simulate_window(gesture, user=None, rng=None):
    """Return one (WINDOW, 11) array of simulated sensor readings.

    `user` is a dict of per-wearer quirks (hand size, sensor offsets, speed,
    noise) so that models are tested on people they were not trained on.
    """
    rng = rng or np.random.default_rng()
    user = user or make_user(rng)
    spec = dict(GESTURE_SPEC[gesture])
    # Half-formed signs: ~25% of the time the hand drifts partway toward another
    # gesture (lazy / rushed signing). This is what makes similar signs confusable.
    if rng.random() < 0.25:
        other = GESTURE_SPEC[rng.choice([g for g in GESTURES if g != gesture])]
        b = rng.uniform(0.25, 0.65)
        spec["flex"] = list((1 - b) * np.array(spec["flex"]) + b * np.array(other["flex"]))
        spec["tilt"] = list((1 - b) * np.array(spec["tilt"]) + b * np.array(other["tilt"]))
        spec["motion"] = [(a, amp * (1 - b), f) for a, amp, f in spec["motion"]] + \
                         [(a, amp * b, f) for a, amp, f in other["motion"]]
    t = np.arange(WINDOW) / FS
    x = np.zeros((WINDOW, 11))

    # --- flex sensors: per-repetition sloppiness + user gain/offset + noise + drift
    flex = np.array(spec["flex"]) + rng.normal(0, user["flex_sloppy"], 5)
    flex = flex * user["flex_gain"] + user["flex_offset"]
    drift = np.linspace(0, rng.normal(0, user["drift"]), WINDOW)[:, None]
    x[:, :5] = flex[None, :] + drift + rng.normal(0, user["flex_noise"], (WINDOW, 5))

    # --- IMU: gravity vector (hand orientation) + periodic motion + noise
    tilt = np.array(spec["tilt"]) + rng.normal(0, user["tilt_sloppy"], 3)
    x[:, 5:8] = tilt[None, :]
    phase = rng.uniform(0, 2 * np.pi)
    speed = user["speed"] * rng.uniform(0.9, 1.1)
    amp_scale = user["amp"] * rng.uniform(0.85, 1.15)
    for axis, amp, freq in spec["motion"]:
        x[:, 5 + axis] += amp * amp_scale * np.sin(2 * np.pi * freq * speed * t + phase)
    x[:, 5:8] += rng.normal(0, 0.03, (WINDOW, 3))
    x[:, 8:11] += rng.normal(0, 0.6, (WINDOW, 3))
    return x


def make_user(rng):
    """Random wearer: different hand size, speed, sensor bias and noise."""
    return dict(
        flex_gain=rng.uniform(0.7, 1.3),
        flex_offset=rng.normal(0, 0.08),
        flex_sloppy=rng.uniform(0.10, 0.28),
        flex_noise=rng.uniform(0.03, 0.08),
        drift=rng.uniform(0.0, 0.15),
        tilt_sloppy=rng.uniform(0.2, 0.55),
        speed=rng.uniform(0.6, 1.5),
        amp=rng.uniform(0.4, 1.6),
    )


def simulate_noise_window(rng=None):
    """A window that is *not* one of the 5 gestures (random flailing / idle hand).
    Used only to test that the confidence check rejects unknown input."""
    rng = rng or np.random.default_rng()
    x = np.zeros((WINDOW, 11))
    x[:, :5] = rng.uniform(0.0, 1.0, 5)[None, :] + rng.normal(0, 0.03, (WINDOW, 5))
    x[:, 5:8] = rng.normal(0, 0.6, 3)[None, :] + rng.normal(0, 0.05, (WINDOW, 3))
    t = np.arange(WINDOW) / FS
    for a in range(6):
        x[:, 5 + a] += rng.uniform(0, 2.5) * np.sin(2 * np.pi * rng.uniform(0.3, 8) * t + rng.uniform(0, 6))
    return x


def generate_dataset(n_users=20, reps_per_gesture=40, seed=0):
    """Returns X (n, WINDOW, 11), y (n,), users (n,) - 'users' is for group-wise CV."""
    rng = np.random.default_rng(seed)
    X, y, users = [], [], []
    for u in range(n_users):
        user = make_user(rng)
        for g_idx, g in enumerate(GESTURES):
            for _ in range(reps_per_gesture):
                X.append(simulate_window(g, user, rng))
                y.append(g_idx)
                users.append(u)
    return np.array(X), np.array(y), np.array(users)
