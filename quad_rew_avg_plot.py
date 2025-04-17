import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize lists for rewards and steps
lppo_rew, lppo_steps = [], []
ly_rew, ly_steps = [], []
ppo_rew, ppo_steps = [], []

# Load rewards and step arrays for 10 seeds for each algorithm
for i in range(1, 11):
    lppo_rew.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "seed_" + str(i), "returns.npy")))
    lppo_steps.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "seed_" + str(i), "steps.npy")))

    ly_rew.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_ly", "seed_" + str(i), "returns.npy")))
    ly_steps.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_ly", "seed_" + str(i), "steps.npy")))

    ppo_rew.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_ppo", "seed_" + str(i), "returns.npy")))
    ppo_steps.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_ppo", "seed_" + str(i), "steps.npy")))

# Define bin size (e.g., 1000 steps per bin)
bin_size = 1000

# Function to aggregate rewards by bins
def aggregate_rewards_by_bins(steps_list, rewards_list, bin_size):
    reward_by_bin = defaultdict(list)
    for steps, rewards in zip(steps_list, rewards_list):
        for step, reward in zip(steps, rewards):
            bin_index = step // bin_size  # Determine the bin index
            reward_by_bin[bin_index].append(reward)
    return reward_by_bin

# Aggregate rewards for each algorithm
lppo_reward_by_bin = aggregate_rewards_by_bins(lppo_steps, lppo_rew, bin_size)
ly_reward_by_bin = aggregate_rewards_by_bins(ly_steps, ly_rew, bin_size)
ppo_reward_by_bin = aggregate_rewards_by_bins(ppo_steps, ppo_rew, bin_size)

# Function to compute mean and std for bins
def compute_mean_std(reward_by_bin):
    sorted_bins = sorted(reward_by_bin.keys())
    mean_rewards = [np.mean(reward_by_bin[bin_index]) for bin_index in sorted_bins]
    std_rewards = [np.std(reward_by_bin[bin_index]) for bin_index in sorted_bins]
    bin_centers = [bin_index * bin_size + bin_size // 2 for bin_index in sorted_bins]
    return bin_centers, mean_rewards, std_rewards

# Compute mean, std, and bin centers for each algorithm
lppo_bin_centers, lppo_mean_rewards, lppo_std_rewards = compute_mean_std(lppo_reward_by_bin)
ly_bin_centers, ly_mean_rewards, ly_std_rewards = compute_mean_std(ly_reward_by_bin)
ppo_bin_centers, ppo_mean_rewards, ppo_std_rewards = compute_mean_std(ppo_reward_by_bin)

def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

lppo_mean_rewards = moving_average(lppo_mean_rewards, window_size=50)
lppo_std_rewards = moving_average(lppo_std_rewards, window_size=50)
ly_mean_rewards = moving_average(ly_mean_rewards, window_size=50)
ly_std_rewards = moving_average(ly_std_rewards, window_size=50)
ppo_mean_rewards = moving_average(ppo_mean_rewards, window_size=50)
ppo_std_rewards = moving_average(ppo_std_rewards, window_size=50)

# Plot the mean with shaded standard deviation for all algorithms
plt.figure(figsize=(10, 6))

# LPPO
plt.plot(lppo_bin_centers, lppo_mean_rewards, label="LPPO Mean", color="blue")
plt.fill_between(lppo_bin_centers, 
                 np.array(lppo_mean_rewards) - np.array(lppo_std_rewards), 
                 np.array(lppo_mean_rewards) + np.array(lppo_std_rewards), 
                 color="blue", alpha=0.2)

# LY
plt.plot(ly_bin_centers, ly_mean_rewards, label="LY Mean", color="green")
plt.fill_between(ly_bin_centers, 
                 np.array(ly_mean_rewards) - np.array(ly_std_rewards), 
                 np.array(ly_mean_rewards) + np.array(ly_std_rewards), 
                 color="green", alpha=0.2)

# PPO
plt.plot(ppo_bin_centers, ppo_mean_rewards, label="PPO Mean", color="red")
plt.fill_between(ppo_bin_centers, 
                 np.array(ppo_mean_rewards) - np.array(ppo_std_rewards), 
                 np.array(ppo_mean_rewards) + np.array(ppo_std_rewards), 
                 color="red", alpha=0.2)

# Add labels, title, and legend
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Quadrotor Rewards")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()