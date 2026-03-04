import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import os
import onnxruntime as ort
import time

API_URL = "https://sev-guardian1.onrender.com/score"

st.set_page_config(
    page_title="SEV Guardian SOC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- GET MODEL FEATURE SIZE ----------------
session = ort.InferenceSession("models/mlp.onnx")
input_shape = session.get_inputs()[0].shape
N_FEATURES = input_shape[1]

# ---------------- ATTACK INTELLIGENCE ----------------
def classify_attack(score):
    if score < 0.4:
        return "Normal Traffic"
    elif score < 0.6:
        return "Reconnaissance / Port Scan"
    elif score < 0.8:
        return "DDoS / Brute Force Pattern"
    else:
        return "Advanced Persistent Threat (APT)"

# ---------------- HEADER ----------------
st.markdown("""
# 🔐 SEV Guardian  
### Hardware-Accelerated AI Threat Intelligence Platform
""")

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Control Panel")
simulate = st.sidebar.toggle("Simulate Traffic")
benchmark = st.sidebar.toggle("Run Benchmark")

# ---------------- SESSION STATE ----------------
if "scores" not in st.session_state:
    st.session_state.scores = []

# ---------------- CALL API ----------------
def call_api():
    features = np.random.rand(N_FEATURES).tolist()
    r = requests.post(API_URL, json={"features": features})
    return r.json()["threat_score"]

# ---------------- MAIN LOGIC ----------------
if simulate:
    score = call_api()
    st.session_state.scores.append(score)

    severity = "Low"
    if score > 0.7:
        severity = "High"
    elif score > 0.4:
        severity = "Medium"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Threat Score", round(score, 4))

    with col2:
        st.metric("Severity Level", severity)

    with col3:
        st.metric("Events Processed", len(st.session_state.scores))

    # ---------------- ATTACK INTELLIGENCE PANEL ----------------
    attack_type = classify_attack(score)

    st.markdown("### 🧠 Threat Intelligence Analysis")
    st.info(f"Detected Pattern: {attack_type}")

    if attack_type != "Normal Traffic":
        st.warning("Recommended Action: Enable firewall rate limiting and investigate suspicious IP ranges.")

    # ---------------- THREAT GAUGE ----------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Threat Level"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'thickness': 0.3},
            'steps': [
                {'range': [0, 0.4], 'color': "green"},
                {'range': [0.4, 0.7], 'color': "orange"},
                {'range': [0.7, 1], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

# ---------------- THREAT TIMELINE ----------------
if st.session_state.scores:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.scores,
        mode="lines+markers",
        line=dict(width=3)
    ))
    fig.update_layout(
        title="Live Threat Score Timeline",
        xaxis_title="Event",
        yaxis_title="Threat Score"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- MODEL EVALUATION ----------------
st.markdown("---")
st.subheader("📊 Model Evaluation")

if os.path.exists("models/roc_curve.png"):
    colA, colB = st.columns(2)
    with colA:
        st.image("models/roc_curve.png", caption="ROC Curve")
    with colB:
        st.image("models/confusion_matrix.png", caption="Confusion Matrix")

# ---------------- BENCHMARK PANEL ----------------
if benchmark:
    st.markdown("---")
    st.subheader("⚡ Performance Benchmark")

    start = time.time()
    for _ in range(100):
        call_api()
    end = time.time()

    avg_time = (end - start)/100 * 1000
    st.success(f"Average Inference Time: {round(avg_time,2)} ms")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Powered by ONNX Runtime | ROCm Compatible | AMD Optimized Architecture")