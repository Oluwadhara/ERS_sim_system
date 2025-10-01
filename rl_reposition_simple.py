import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# ENV parameters
Z = 5  # grid ZxZ zones
NUM_ZONES = Z*Z
LAMBDA = 0.5  # avg calls per timestep across whole city
TIMESTEPS = 2000

def zone_id_from_xy(x,y):
    return x*Z + y

def sample_call_zone():
    return random.randint(0, NUM_ZONES-1)

class SimpleEnv:
    def __init__(self):
        self.ambulance_zone = NUM_ZONES//2  # start centrally
        self.time = 0

    def reset(self):
        self.ambulance_zone = NUM_ZONES//2
        self.time = 0
        return (self.ambulance_zone,)

    def step(self, action):
        self.ambulance_zone = action
        n_calls = np.random.poisson(LAMBDA)
        total_response = 0.0
        for _ in range(n_calls):
            cz = sample_call_zone()
            ax, ay = divmod(self.ambulance_zone, Z)
            cx, cy = divmod(cz, Z)
            dist = abs(ax-cx) + abs(ay-cy)
            response_time = dist + 1.0
            total_response += response_time
        reward = - (total_response / (n_calls if n_calls > 0 else 1))
        self.time += 1
        done = (self.time >= TIMESTEPS)
        state = (self.ambulance_zone,)
        return state, reward, done

# Q-learning
Q = defaultdict(lambda: np.zeros(NUM_ZONES))
alpha = 0.1
gamma = 0.99
epsilon = 0.2
env = SimpleEnv()
episodes = 5000

rewards_per_episode = []  # <-- NEW

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if random.random() < epsilon:
            action = random.randrange(NUM_ZONES)
        else:
            action = np.argmax(Q[state])
        next_state, reward, done = env.step(action)
        best_next = np.max(Q[next_state])
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * best_next - Q[state][action])
        state = next_state
        total_reward += reward
    rewards_per_episode.append(total_reward)  # <-- store total reward

# After training: evaluate greedy policy
env = SimpleEnv()
state = env.reset()
tot_reward = 0
for _ in range(1000):
    action = np.argmax(Q[state])
    state, reward, done = env.step(action)
    tot_reward += reward
print("Average reward (eval):", tot_reward/1000.0)

# Plot learning curve
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("RL Training Performance (Q-learning)")
plt.grid(True)
plt.show()
