import matplotlib.pyplot as plt
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

ppo_rew = np.load(os.path.join(curr_dir, "server-files", "quad-track", "ppo-rt", "returns.npy"))
ly_rew = np.load(os.path.join(curr_dir, "server-files", "quad-track", "ly-rt", "returns.npy"))
lppo_rew = np.load(os.path.join(curr_dir, "server-files", "quad-track", "lppo-rt", "returns.npy"))

ppo_steps = np.load(os.path.join(curr_dir, "server-files", "quad-track", "ppo-rt", "steps.npy"))
lppo_steps = np.load(os.path.join(curr_dir, "server-files", "quad-track", "lppo-rt", "steps.npy"))
ly_steps = np.load(os.path.join(curr_dir, "server-files", "quad-track", "ly-rt", "steps.npy"))

# Function to compute the moving average
def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

# Compute the moving averages
ppo_avg = moving_average(ppo_rew, window_size=50)
ly_avg = moving_average(ly_rew, window_size=50)
lppo_avg = moving_average(lppo_rew, window_size=50)

# Plot the moving averages
plt.figure(figsize=(10, 6))
plt.plot(ppo_steps, ppo_avg, label="PPO Rewards (Moving Avg)", color="blue")
plt.plot(ly_steps, ly_avg, label="LY Rewards (Moving Avg)", color="green")
plt.plot(lppo_steps, lppo_avg,label="LPPO Rewards (Moving Avg)", color="red")

# Add labels, title, and legend
plt.xlabel("Steps")
plt.ylabel("Average Rewards (Last 50 Episodes)")
plt.title("Reward Comparison Across Algorithms (Moving Average)")
plt.legend()

# plt.xlim(0, 1.75e7)
# Show the plot
plt.show()