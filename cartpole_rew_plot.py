import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize lists for rewards and steps
lsac_rew, lsac_steps = [], []
ly_rew, ly_steps = [], []
ppo_rew, ppo_steps = [], []
sac_rew, sac_steps = [], []

# Load rewards and step arrays for 10 seeds for each algorithm
for i in range(1, 11):
    lsac_rew.append(np.load(os.path.join(curr_dir, "data", "CustomCart-v1", "lsac", "seed_" + str(i), "reward_arr.npy")))
    lsac_steps.append(np.load(os.path.join(curr_dir, "data", "CustomCart-v1", "lsac", "seed_" + str(i), "step_arr.npy")))

    ly_rew.append(np.load(os.path.join(curr_dir, "data", "CustomCart-v1", "ly", "seed_" + str(i), "reward_arr.npy")))
    ly_steps.append(np.load(os.path.join(curr_dir, "data", "CustomCart-v1", "ly", "seed_" + str(i), "step_arr.npy")))

    ppo_rew.append(np.load(os.path.join(curr_dir, "data", "CustomCart-v1", "ppo", "seed_" + str(i), "reward_arr.npy")))
    ppo_steps.append(np.load(os.path.join(curr_dir, "data", "CustomCart-v1", "ppo", "seed_" + str(i), "step_arr.npy")))

    sac_rew.append(np.load(os.path.join(curr_dir, "data", "CustomCart-v1", "sac", "seed_" + str(i), "reward_arr.npy")))
    sac_steps.append(np.load(os.path.join(curr_dir, "data", "CustomCart-v1", "sac", "seed_" + str(i), "step_arr.npy")))

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
lsac_reward_by_bin = aggregate_rewards_by_bins(lsac_steps, lsac_rew, bin_size)
ly_reward_by_bin = aggregate_rewards_by_bins(ly_steps, ly_rew, bin_size)
ppo_reward_by_bin = aggregate_rewards_by_bins(ppo_steps, ppo_rew, bin_size)
sac_reward_by_bin = aggregate_rewards_by_bins(sac_steps, sac_rew, bin_size)

# Function to compute mean and std for bins
def compute_mean_std(reward_by_bin):
    sorted_bins = sorted(reward_by_bin.keys())
    mean_rewards = [np.mean(reward_by_bin[bin_index]) for bin_index in sorted_bins]
    std_rewards = [np.std(reward_by_bin[bin_index]) for bin_index in sorted_bins]
    bin_centers = [bin_index * bin_size + bin_size // 2 for bin_index in sorted_bins]
    return bin_centers, mean_rewards, std_rewards

# Compute mean, std, and bin centers for each algorithm
lsac_bin_centers, lsac_mean_rewards, lsac_std_rewards = compute_mean_std(lsac_reward_by_bin)
ly_bin_centers, ly_mean_rewards, ly_std_rewards = compute_mean_std(ly_reward_by_bin)
ppo_bin_centers, ppo_mean_rewards, ppo_std_rewards = compute_mean_std(ppo_reward_by_bin)
sac_bin_centers, sac_mean_rewards, sac_std_rewards = compute_mean_std(sac_reward_by_bin)

def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

lsac_mean_rewards = moving_average(lsac_mean_rewards, window_size=50)
lsac_std_rewards = moving_average(lsac_std_rewards, window_size=50)
ly_mean_rewards = moving_average(ly_mean_rewards, window_size=50)
ly_std_rewards = moving_average(ly_std_rewards, window_size=50)
ppo_mean_rewards = moving_average(ppo_mean_rewards, window_size=50)
ppo_std_rewards = moving_average(ppo_std_rewards, window_size=50)
sac_mean_rewards = moving_average(sac_mean_rewards, window_size=50)
sac_std_rewards = moving_average(sac_std_rewards, window_size=50)

# Plot the mean with shaded standard deviation for all algorithms
plt.figure(figsize=(10, 6))

# LPPO
plt.plot(lsac_bin_centers, lsac_mean_rewards, label="LPPO Mean", color="blue")
plt.fill_between(lsac_bin_centers, 
                 np.array(lsac_mean_rewards) - np.array(lsac_std_rewards), 
                 np.array(lsac_mean_rewards) + np.array(lsac_std_rewards), 
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

# SAC
plt.plot(sac_bin_centers, sac_mean_rewards, label="SAC Mean", color="orange")
plt.fill_between(sac_bin_centers, 
                 np.array(sac_mean_rewards) - np.array(sac_std_rewards), 
                 np.array(sac_mean_rewards) + np.array(sac_std_rewards), 
                 color="orange", alpha=0.2)

# Add labels, title, and legend
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Quadrotor Rewards")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()