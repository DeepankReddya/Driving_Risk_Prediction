from rl.env import DrivingEnv
from rl.agent import QLearningAgent
import pickle

env = DrivingEnv()
agent = QLearningAgent()

episodes = 500

for ep in range(episodes):

    state = env.reset()
    state = agent.discretize(state)

    total_reward = 0

    for step in range(20):

        action = agent.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        next_state = agent.discretize(next_state)

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    agent.decay_epsilon()

    if (ep+1) % 50 == 0:
        print(f"Episode {ep+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

# -----------------------------
# SAVE TRAINED AGENT
# -----------------------------
with open("rl_agent.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

print("\n✅ Training complete & agent saved!")