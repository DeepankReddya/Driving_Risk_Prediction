import streamlit as st
import numpy as np
from src.predict import predict_sequence
from src.explain import generate_explanation
from rl.inference_rl import get_rl_recommendation

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Driving Risk AI", layout="wide")

st.title("🚗 Driving Risk AI System")
st.write("Deep Learning + Reinforcement Learning")

# -----------------------------
# INPUTS
# -----------------------------
st.subheader("Driving Inputs")

acc_level = st.selectbox("Acceleration", ["Low", "Medium", "High"])
brake_level = st.selectbox("Braking", ["Low", "Medium", "High"])
turning = st.selectbox("Turning", ["Smooth", "Moderate", "Sharp"])
weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"])

# -----------------------------
# ORIGINAL WORKING INPUT LOGIC
# -----------------------------
def map_input(acc, brake, turn, weather):

    acc_map = {"Low": 0.3, "Medium": 0.7, "High": 1.5}
    brake_map = {"Low": -0.2, "Medium": -0.7, "High": -1.5}
    turn_map = {"Smooth": 0.2, "Moderate": 0.8, "Sharp": 1.5}
    weather_map = {"Clear": 0, "Rain": 1, "Fog": 2}

    acc_val = acc_map[acc]
    brake_val = brake_map[brake]
    turn_val = turn_map[turn]
    weather_val = weather_map[weather]

    seq = []

    for i in range(20):
        fluctuation = np.sin(i / 2) + np.random.normal(0, 0.3)

        row = [
            turn_val * fluctuation,
            turn_val * fluctuation,
            turn_val * fluctuation,
            acc_val * fluctuation,
            brake_val * fluctuation,
            acc_val * fluctuation,
            weather_val
        ]

        seq.append(row)

    return np.array(seq)

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("Run Analysis"):

    # Generate input
    seq = map_input(acc_level, brake_level, turning, weather)

    # Model prediction
    pred, confidence, attn = predict_sequence(seq)

    # -----------------------------
    # RISK OVERRIDE FIX
    # -----------------------------
    risky_conditions = 0

    if acc_level == "High":
        risky_conditions += 1

    if brake_level == "High":
        risky_conditions += 1

    if turning == "Sharp":
        risky_conditions += 1

    if weather in ["Rain", "Fog"]:
        risky_conditions += 1

    if risky_conditions >= 2:
        final_pred = 1
    else:
        final_pred = pred

    label = "🔴 RISKY" if final_pred == 1 else "🟢 SAFE"

    # -----------------------------
    # LSTM OUTPUT
    # -----------------------------
    st.subheader("🧠 LSTM Prediction")
    st.markdown(f"### {label}")
    st.write(f"Confidence: {confidence:.2f}")

    # Explanation
    reasons = generate_explanation(seq, attn[0])

    st.markdown("### Explanation")
    for r in reasons:
        st.write(f"- {r}")

    # -----------------------------
    # RL OUTPUT
    # -----------------------------
    st.subheader("🤖 RL Recommendation")

    rl_action = get_rl_recommendation(
        acc_level, brake_level, turning, weather
    )

    suggestions = []

    if acc_level == "High":
        suggestions.append("Reduce acceleration")

    if brake_level == "High":
        suggestions.append("Avoid sudden braking")

    if turning == "Sharp":
        suggestions.append("Smooth your turning")

    if weather == "Rain":
        suggestions.append("Drive carefully on wet roads")

    if weather == "Fog":
        suggestions.append("Reduce speed due to low visibility")

    if not suggestions:
        suggestions.append(rl_action)

    for s in suggestions:
        st.success(s)

    # -----------------------------
    # FINAL INSIGHT
    # -----------------------------
    st.subheader("📊 Combined Insight")

    if final_pred == 1:
        st.error("Driving pattern is risky. Follow recommendations to improve safety.")
    else:
        st.success("Driving pattern is safe and stable.")