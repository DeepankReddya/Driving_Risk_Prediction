import pickle
import numpy as np
from rl.agent import QLearningAgent

agent = QLearningAgent()

with open("rl_agent.pkl", "rb") as f:
    agent.q_table = pickle.load(f)

agent.epsilon = 0


# -----------------------------
# INPUT → STATE
# -----------------------------
def map_to_state(acc, brake, turn, weather):

    acc_map = {"Low": 0.2, "Medium": 0.6, "High": 1.0}
    brake_map = {"Low": 0.2, "Medium": 0.6, "High": 1.0}
    turn_map = {"Smooth": 0.2, "Moderate": 0.6, "Sharp": 1.0}
    weather_map = {"Clear": 0, "Rain": 1, "Fog": 2}

    return np.array([
        acc_map[acc],
        brake_map[brake],
        turn_map[turn],
        weather_map[weather]
    ])


# -----------------------------
# RULE-BASED SAFETY (KEY FIX)
# -----------------------------
def rule_based_action(acc, brake, turn, weather):

    if acc == "High":
        return "Reduce acceleration"

    if brake == "High":
        return "Avoid harsh braking"

    if turn == "Sharp":
        return "Smooth your turning"

    if weather == "Rain":
        return "Drive cautiously on wet roads"

    if weather == "Fog":
        return "Reduce speed due to low visibility"

    return None


# -----------------------------
# RL + RULE HYBRID
# -----------------------------
def get_rl_recommendation(acc, brake, turn, weather):

    # 1️⃣ Rule-based safety check
    rule_action = rule_based_action(acc, brake, turn, weather)

    # 2️⃣ RL suggestion
    state = map_to_state(acc, brake, turn, weather)
    state_disc = agent.discretize(state)

    q_values = [agent.get_q(state_disc, a) for a in agent.actions]
    rl_action_idx = int(np.argmax(q_values))

    action_map = {
        0: "Maintain current driving",
        1: "Reduce acceleration",
        2: "Apply controlled braking",
        3: "Smooth your turning"
    }

    rl_action = action_map[rl_action_idx]

    # -----------------------------
    # FINAL DECISION (IMPORTANT)
    # -----------------------------
    if rule_action is not None:
        return rule_action   # override RL if risky

    return rl_action