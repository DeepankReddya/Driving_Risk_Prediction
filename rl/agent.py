import numpy as np
import random

class QLearningAgent:

    def __init__(self):
        self.actions = [0, 1, 2, 3]

        # Q-table: dictionary
        self.q_table = {}

        self.alpha = 0.1      # learning rate
        self.gamma = 0.9      # discount factor
        self.epsilon = 1.0    # exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    # -----------------------------
    # DISCRETIZE STATE
    # -----------------------------
    def discretize(self, state):
        return tuple((state * 10).astype(int))

    # -----------------------------
    # GET Q VALUE
    # -----------------------------
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    # -----------------------------
    # CHOOSE ACTION (ε-greedy)
    # -----------------------------
    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = [self.get_q(state, a) for a in self.actions]
        return int(np.argmax(q_values))

    # -----------------------------
    # UPDATE Q-TABLE
    # -----------------------------
    def update(self, state, action, reward, next_state):

        max_next_q = max([self.get_q(next_state, a) for a in self.actions])

        old_q = self.get_q(state, action)

        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

        self.q_table[(state, action)] = new_q

    # -----------------------------
    # DECAY EXPLORATION
    # -----------------------------
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay