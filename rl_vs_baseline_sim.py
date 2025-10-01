import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Load dataset
df = pd.read_csv("synthetic_calls.csv")

# --- Fix column names ---
if "timeReceived" in df.columns:
    df = df.sort_values("timeReceived").reset_index(drop=True)
elif "Timestamp" in df.columns:
    # convert Timestamp to datetime and then numeric
    df["timeReceived"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("timeReceived").reset_index(drop=True)
else:
    # fallback: just use index as time
    df["timeReceived"] = range(len(df))

# Extract call data
calls = df[["timeReceived", "X", "Y", "Severity"]].values

# Grid setup
GRID_SIZE = 100
AMB_SPEED = 1.0  # units per timestep

# RL parameters
Z = 5  # 5x5 zones
NUM_ZONES = Z * Z
alpha = 0.1
gamma = 0.95
epsilon = 0.2
episodes = 300

def zone_id(x, y):
    return (x // (GRID_SIZE // Z)) * Z + (y // (GRID_SIZE // Z))

# Environment
class ERSEnv:
    def __init__(self, calls):
        self.calls = calls
        self.reset()

    def reset(self):
        self.t = 0
        self.ambulance_zone = NUM_ZONES // 2
        self.done = False
        return (self.ambulance_zone,)

    def step(self, action):
        self.ambulance_zone = action
        # process call if within timeline
        if self.t < len(self.calls):
            _, x, y, sev = self.calls[self.t]
            cx, cy = divmod(self.ambulance_zone, Z)
            zx = (GRID_SIZE // Z) * cx + (GRID_SIZE // (2*Z))
            zy = (GRID_SIZE // Z) * cy + (GRID_SIZE // (2*Z))
            dist = abs(zx - x) + abs(zy - y)
            response_time = dist / AMB_SPEED
            reward = -response_time
        else:
            reward = 0
        self.t += 1
        if self.t >= len(self.calls):
            self.done = True
        return (self.ambulance_zone,), reward, self.done

# Q-learning
Q = defaultdict(lambda: np.zeros(NUM_ZONES))
env = ERSEnv(calls)
rewards_per_episode = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        if random.random() < epsilon:
            action = random.randrange(NUM_ZONES)
        else:
            action = np.argmax(Q[state])
        next_state, reward, done = env.step(action)
        total_reward += reward
        best_next = np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
        state = next_state
    rewards_per_episode.append(total_reward)

# Evaluate baseline (always central zone)
def baseline_policy(calls):
    total_rt = 0
    for _, x, y, sev in calls:
        cx, cy = divmod(NUM_ZONES // 2, Z)
        zx = (GRID_SIZE // Z) * cx + (GRID_SIZE // (2*Z))
        zy = (GRID_SIZE // Z) * cy + (GRID_SIZE // (2*Z))
        dist = abs(zx - x) + abs(zy - y)
        total_rt += dist / AMB_SPEED
    return total_rt / len(calls)

baseline_avg = baseline_policy(calls)

# Evaluate RL policy
def rl_policy(calls, Q):
    env = ERSEnv(calls)
    state = env.reset()
    total_rt = 0
    count = 0
    while not env.done:
        action = np.argmax(Q[state])
        next_state, reward, done = env.step(action)
        total_rt += -reward  # since reward = -response time
        count += 1
        state = next_state
    return total_rt / count

rl_avg = rl_policy(calls, Q)

print(f"Baseline average response time: {baseline_avg:.2f}")
print(f"RL average response time: {rl_avg:.2f}")

# --- Plots ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("RL Training Performance")

plt.subplot(1,2,2)
plt.bar(["Baseline", "RL"], [baseline_avg, rl_avg], color=["gray", "green"])
plt.ylabel("Avg Response Time")
plt.title("Baseline vs RL")
plt.show()
