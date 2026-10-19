"""Streamlit simulation of the smart glove. Run:  streamlit run app.py"""
import numpy as np, streamlit as st, matplotlib.pyplot as plt
from glove_sim.gestures import GESTURES, CHANNELS, simulate_window, simulate_noise_window, make_user, FS
from glove_sim.model import GloveClassifier
from glove_sim.speech import speak, phrase, LANGS

st.set_page_config(page_title="Smart Glove Simulator", page_icon="🧤", layout="wide")
st.title("🧤 Adaptive Multilingual Smart Glove - Simulation")
st.caption("Group 17 · Simulated flex-sensor + MPU6050 data → ML classifier → confidence check → speech")

@st.cache_resource
def load(): return GloveClassifier()
clf = load()

with st.sidebar:
    lang = st.selectbox("Output language", list(LANGS), format_func=lambda k: LANGS[k])
    gesture = st.selectbox("Gesture performed by wearer", GESTURES + ["random movement (unknown)"])
    wearer = st.number_input("Wearer ID (each ID = a different simulated hand)", 0, 999, 1)
    thr = st.slider("Confidence threshold", 0.3, 0.95, float(clf.threshold), 0.05)
    go = st.button("Perform gesture", type="primary", use_container_width=True)
clf.threshold = thr

if go:
    rng = np.random.default_rng()
    user = make_user(np.random.default_rng(int(wearer)))
    w = simulate_noise_window(rng) if gesture.startswith("random") else simulate_window(gesture, user, rng)
    pred, conf, probs = clf.predict_window(w)
    t = np.arange(len(w)) / FS

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3))
        for i in range(5): ax.plot(t, w[:, i], label=CHANNELS[i].replace("flex_", ""))
        ax.set_title("Flex sensors (0 = straight, 1 = bent)"); ax.set_xlabel("s"); ax.legend(ncol=5, fontsize=7)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3))
        for i in range(5, 11): ax.plot(t, w[:, i], label=CHANNELS[i])
        ax.set_title("MPU6050 accel (g) + gyro (rad/s)"); ax.set_xlabel("s"); ax.legend(ncol=6, fontsize=7)
        st.pyplot(fig)

    st.subheader("Classifier confidence")
    st.bar_chart({g: float(p) for g, p in zip(GESTURES, probs)})
    if pred is None:
        st.warning(f"⚠️ Low confidence ({conf:.0%} < {thr:.0%}). Glove vibrates - no speech produced. Please retry.")
    else:
        st.success(f"Recognised **{pred}** ({conf:.0%})  →  “{phrase(pred, lang)}”")
        audio = speak(pred, lang)
        if audio: st.audio(audio, autoplay=True)
        else: st.info("Audio needs internet (gTTS). Text shown above.")
        if pred == "emergency":
            st.error("🚨 SOS triggered: GPS location + alert sent to registered contacts (simulated).")
else:
    st.info("Pick a gesture in the sidebar and press **Perform gesture**.")
