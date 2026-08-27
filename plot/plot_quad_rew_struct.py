import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize lists for rewards and steps
rew, steps = [], []

# Load rewards and step arrays for 10 seeds for each algorithm
for i in range(1, 11):
    rew.append(np.load(os.path.join(curr_dir, "data", "struct", "Quadrotor", "beta_0.5", "mu_0.1", "seed_" + str(i), "returns.npy")))
    steps.append(np.load(os.path.join(curr_dir, "data", "struct", "Quadrotor", "beta_0.5", "mu_0.1", "seed_" + str(i), "steps.npy")))

# Define bin size (e.g., 1000 steps per bin)
bin_size = 500

# Function to aggregate rewards by bins
def aggregate_rewards_by_bins(steps_list, rewards_list, bin_size):
    reward_by_bin = defaultdict(list)
    for steps, rewards in zip(steps_list, rewards_list):
        for step, reward in zip(steps, rewards):
            bin_index = step // bin_size  # Determine the bin index
            reward_by_bin[bin_index].append(reward)
    return reward_by_bin

# Aggregate rewards for each algorithm
reward_by_bin = aggregate_rewards_by_bins(steps, rew, bin_size)

# Function to compute mean and std for bins
def compute_mean_std(reward_by_bin):
    sorted_bins = sorted(reward_by_bin.keys())
    mean_rewards = [np.mean(reward_by_bin[bin_index]) for bin_index in sorted_bins]
    std_rewards = [np.std(reward_by_bin[bin_index]) for bin_index in sorted_bins]
    bin_centers = [bin_index * bin_size + bin_size // 2 for bin_index in sorted_bins]
    return bin_centers, mean_rewards, std_rewards

# Compute mean, std, and bin centers for each algorithm
bin_centers, mean_rewards, std_rewards = compute_mean_std(reward_by_bin)

def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

mean_rewards = moving_average(mean_rewards, window_size=50)
std_rewards = moving_average(std_rewards, window_size=50)

# Plot the mean with shaded standard deviation for all algorithms
plt.figure(figsize=(10, 6))

# LPPO
plt.plot(bin_centers, mean_rewards, label="LPPO Mean", color="blue")
plt.fill_between(bin_centers, 
                 np.array(mean_rewards) - np.array(std_rewards), 
                 np.array(mean_rewards) + np.array(std_rewards), 
                 color="blue", alpha=0.2)

# Add labels, title, and legend
plt.xlim(0, 1e7)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Quadrotor Structural Changes NN")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()