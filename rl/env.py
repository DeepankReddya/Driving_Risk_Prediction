import numpy as np

class DrivingEnv:

    def __init__(self):
        self.max_steps = 20
        self.reset()

    def reset(self):
        self.steps = 0

        self.state = np.array([
            np.random.uniform(0, 1),   # acceleration
            np.random.uniform(0, 1),   # braking
            np.random.uniform(0, 1),   # turning
            np.random.choice([0, 1, 2])  # weather
        ], dtype=float)

        return self.state

    def step(self, action):
        acc, brake, turn, weather = self.state

        # -----------------------------
        # ACTIONS
        # -----------------------------
        if action == 1:
            acc = max(0, acc - 0.2)

        elif action == 2:
            brake = min(1, brake + 0.2)

        elif action == 3:
            turn = max(0, turn - 0.2)

        # -----------------------------
        # ADD SMALL RANDOMNESS (REALISM)
        # -----------------------------
        acc += np.random.normal(0, 0.02)
        brake += np.random.normal(0, 0.02)
        turn += np.random.normal(0, 0.02)

        acc = np.clip(acc, 0, 1)
        brake = np.clip(brake, 0, 1)
        turn = np.clip(turn, 0, 1)

        # -----------------------------
        # RISK CALCULATION
        # -----------------------------
        risk = 0

        if acc > 0.8: risk += 2
        if brake > 0.8: risk += 2
        if turn > 0.8: risk += 2
        if weather == 1: risk += 1
        if weather == 2: risk += 2

        # -----------------------------
        # IMPROVED REWARD
        # -----------------------------
        if risk >= 5:
            reward = -10
        elif risk >= 3:
            reward = -3
        else:
            reward = +3

        # -----------------------------
        # STEP COUNT
        # -----------------------------
        self.steps += 1
        done = self.steps >= self.max_steps

        self.state = np.array([acc, brake, turn, weather])

        return self.state, reward, done, {}