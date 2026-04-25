import numpy as np

def generate_explanation(seq, attention):

    reasons = []

    # -----------------------------
    # FEATURE EXTRACTION
    # -----------------------------
    gyro_mean = np.mean(np.abs(seq[:, 0:3]))
    acc_mean = np.mean(np.abs(seq[:, 3:6]))
    min_brake = np.min(seq[:, 4])
    weather_val = int(seq[0, 6])

    high_acc = acc_mean > 0.8
    high_turn = gyro_mean > 0.8
    sudden_brake = min_brake < -0.8

    weather_map = {
        0: "clear weather",
        1: "rainy conditions",
        2: "foggy conditions"
    }

    weather_text = weather_map.get(weather_val, "unknown")

    # -----------------------------
    # RISK LOGIC
    # -----------------------------
    is_risky_behavior = high_acc or high_turn or sudden_brake

    # -----------------------------
    # COMBINED REASONING
    # -----------------------------
    if high_acc and high_turn:
        reasons.append("High acceleration combined with sharp turning increases instability")

    if sudden_brake and high_acc:
        reasons.append("Sudden braking after acceleration indicates unsafe driving")

    if high_turn:
        reasons.append("Frequent or sharp turning detected")

    if high_acc:
        reasons.append("Aggressive acceleration observed")

    if sudden_brake:
        reasons.append("Sudden braking detected")

    # -----------------------------
    # WEATHER
    # -----------------------------
    if weather_val == 1:
        reasons.append("Rain reduces tire grip and increases risk")
    elif weather_val == 2:
        reasons.append("Fog reduces visibility and increases risk")

    # -----------------------------
    # ATTENTION-BASED LOGIC (FIXED)
    # -----------------------------
    peak_step = int(np.argmax(attention))
    peak_value = np.max(attention)

    # Only show if truly meaningful
    if is_risky_behavior and peak_value > 0.15:
        reasons.append(f"Risky driving pattern concentrated around time step {peak_step}")
    elif not is_risky_behavior:
        reasons.append("Driving pattern is smooth and stable")

    # -----------------------------
    # CLEANUP
    # -----------------------------
    reasons = list(set(reasons))

    return reasons