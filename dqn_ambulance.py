import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("synthetic_calls.csv")

# Make sure we have a numeric time
if "Timestamp" in df.columns:
    df["timeReceived"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("timeReceived").reset_index(drop=True)
else:
    df["timeReceived"] = range(len(df))

# Filter only ambulance-related calls
df = df[df["ResponseUnit"] == "Ambulance"].reset_index(drop=True)

# Severity weights
severity_weights = {"Critical": 3.0, "Moderate": 2.0, "Low": 1.0}

# Extract calls
calls = df[["timeReceived", "X", "Y", "Severity"]].values

# -------------------------
# Environment
# -------------------------
GRID_SIZE = 100
AMB_SPEED = 1.0
Z = 5
NUM_ZONES = Z * Z

def zone_center(zone):
    cx, cy = divmod(zone, Z)
    zx = (GRID_SIZE // Z) * cx + (GRID_SIZE // (2*Z))
    zy = (GRID_SIZE // Z) * cy + (GRID_SIZE // (2*Z))
    return zx, zy

class ERSEnv:
    def __init__(self, calls):
        self.calls = calls
        self.reset()

    def reset(self):
        self.t = 0
        self.ambulance_zone = NUM_ZONES // 2
        self.done = False
        return np.array([self.ambulance_zone], dtype=np.float32)

    def step(self, action):
        self.ambulance_zone = int(action)
        if self.t < len(self.calls):
            _, x, y, sev = self.calls[self.t]
            zx, zy = zone_center(self.ambulance_zone)
            dist = abs(zx - x) + abs(zy - y)
            response_time = dist / AMB_SPEED
            weight = severity_weights.get(sev, 1.0)
            reward = -(response_time * weight)
        else:
            reward = 0
        self.t += 1
        if self.t >= len(self.calls):
            self.done = True
        return np.array([self.ambulance_zone], dtype=np.float32), reward, self.done

# -------------------------
# DQN setup
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn(env, episodes=500, gamma=0.95, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001, batch_size=32):
    state_dim = 1
    action_dim = NUM_ZONES
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=5000)

    rewards_log = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.FloatTensor(state))
                    action = torch.argmax(q_vals).item()
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target = rewards + gamma * next_q * (1 - dones)
                loss = nn.MSELoss()(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_log.append(total_reward)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if ep % 20 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {ep}, total reward {total_reward:.2f}, epsilon {epsilon:.2f}")

    return policy_net, rewards_log

# -------------------------
# Train RL
# -------------------------
env = ERSEnv(calls)
policy_net, rewards_log = train_dqn(env, episodes=300)

# -------------------------
# Evaluate RL policy
# -------------------------
def evaluate_policy(env, policy_net):
    state = env.reset()
    total_rt = 0
    count = 0
    while not env.done:
        with torch.no_grad():
            q_vals = policy_net(torch.FloatTensor(state))
            action = torch.argmax(q_vals).item()
        next_state, reward, done = env.step(action)
        total_rt += -reward
        count += 1
        state = next_state
    return total_rt / count

# Baseline
def baseline_policy(calls):
    total_rt = 0
    for _, x, y, sev in calls:
        zx, zy = zone_center(NUM_ZONES // 2)
        dist = abs(zx - x) + abs(zy - y)
        weight = severity_weights.get(sev, 1.0)
        total_rt += dist * weight
    return total_rt / len(calls)

baseline_avg = baseline_policy(calls)
rl_avg = evaluate_policy(ERSEnv(calls), policy_net)

print(f"Baseline avg weighted response time: {baseline_avg:.2f}")
print(f"RL avg weighted response time: {rl_avg:.2f}")

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(rewards_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Performance")

plt.subplot(1,2,2)
plt.bar(["Baseline", "RL"], [baseline_avg, rl_avg], color=["gray","green"])
plt.ylabel("Avg Weighted Response Time")
plt.title("Baseline vs RL")
plt.show()
