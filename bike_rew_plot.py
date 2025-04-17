import matplotlib.pyplot as plt
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

ppo_rew = np.load(os.path.join(curr_dir, "data", "Bicycle-v1", "ppo", "seed_1", "reward_arr.npy"))
ppo_steps = np.load(os.path.join(curr_dir, "data", "Bicycle-v1", "ppo", "seed_1", "step_arr.npy"))

sac_rew = np.load(os.path.join(curr_dir, "data", "Bicycle-v1", "sac", "seed_2", "reward_arr.npy"))
sac_steps = np.load(os.path.join(curr_dir, "data", "Bicycle-v1", "sac", "seed_2", "step_arr.npy"))


# Function to compute the moving average
def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

# Compute the moving averages
ppo_avg = moving_average(ppo_rew, window_size=50)
sac_avg = moving_average(sac_rew, window_size=50)

print(len(ppo_avg))

# Plot the moving averages
# plt.figure(figsize=(10, 6))
# plt.plot(ppo_avg, label="PPO Rewards (Moving Avg)", color="blue")
plt.figure(figsize=(10, 6))
plt.plot(sac_avg, label="SAC)", color="blue")

# Add labels, title, and legend
plt.xlabel("Episodes")
plt.ylabel("Average Rewards (Last 50 Episodes)")
plt.title("Reward Comparison Across Algorithms (Moving Average)")
plt.legend()

# plt.xlim(0, 1.75e7)
# Show the plot
plt.show()