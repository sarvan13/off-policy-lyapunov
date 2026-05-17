import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize lists for rewards and steps
lppo_rew0, lppo_steps0 = [], []
lppo_rew01, lppo_steps01 = [], []
lppo_rew001, lppo_steps001 = [], []
lppo_rew05, lppo_steps05 = [], []
lppo_rew075, lppo_steps075 = [], []
lppo_rew085, lppo_steps085 = [], []
lppo_rew1, lppo_steps1 = [], []
lppo_rew1_1, lppo_steps1_1 = [], []
lppo_rew1_5, lppo_steps1_5 = [], []
ly_rew, ly_steps = [], []
ppo_rew, ppo_steps = [], []


# Load rewards and step arrays for 10 seeds for each algorithm
for i in range(1, 11):
    lppo_rew0.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.0", "seed_" + str(i), "returns.npy")))
    lppo_steps0.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.0", "seed_" + str(i), "steps.npy")))

    lppo_rew01.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.1", "seed_" + str(i), "returns.npy")))
    lppo_steps01.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.1", "seed_" + str(i), "steps.npy")))

    lppo_rew001.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.01", "seed_" + str(i), "returns.npy")))
    lppo_steps001.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.01", "seed_" + str(i), "steps.npy")))

    lppo_rew05.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.5", "seed_" + str(i), "returns.npy")))
    lppo_steps05.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.5", "seed_" + str(i), "steps.npy")))

    lppo_rew075.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.75", "seed_" + str(i), "returns.npy")))
    lppo_steps075.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.75", "seed_" + str(i), "steps.npy")))
    
    if i != 8:  # Skip seed 8 for mu=0.85 due to missing data
        lppo_rew085.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.85", "seed_" + str(i), "returns.npy")))
        lppo_steps085.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_0.85", "seed_" + str(i), "steps.npy")))
    
    lppo_rew1.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_1.0", "seed_" + str(i), "returns.npy")))
    lppo_steps1.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_1.0", "seed_" + str(i), "steps.npy")))

    lppo_rew1_1.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_1.1", "seed_" + str(i), "returns.npy")))
    lppo_steps1_1.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_1.1", "seed_" + str(i), "steps.npy")))

    lppo_rew1_5.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_1.5", "seed_" + str(i), "returns.npy")))
    lppo_steps1_5.append(np.load(os.path.join(curr_dir, "data", "Quadrotor-Still-v1", "clean_lppo", "mu_1.5", "seed_" + str(i), "steps.npy")))
    
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
lppo_reward_by_bin = aggregate_rewards_by_bins(lppo_steps0, lppo_rew0, bin_size)
lppo_reward_by_bin_01 = aggregate_rewards_by_bins(lppo_steps01, lppo_rew01, bin_size)
lppo_reward_by_bin_001 = aggregate_rewards_by_bins(lppo_steps001, lppo_rew001, bin_size)
lppo_reward_by_bin_05 = aggregate_rewards_by_bins(lppo_steps05, lppo_rew05, bin_size)
lppo_reward_by_bin_075 = aggregate_rewards_by_bins(lppo_steps075, lppo_rew075, bin_size)
lppo_reward_by_bin_085 = aggregate_rewards_by_bins(lppo_steps085, lppo_rew085, bin_size)
lppo_reward_by_bin_1 = aggregate_rewards_by_bins(lppo_steps1, lppo_rew1, bin_size)
lppo_reward_by_bin_1_1 = aggregate_rewards_by_bins(lppo_steps1_1, lppo_rew1_1, bin_size)
lppo_reward_by_bin_1_5 = aggregate_rewards_by_bins(lppo_steps1_5, lppo_rew1_5, bin_size)
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
lppo_bin_centers_01, lppo_mean_rewards_01, lppo_std_rewards_01 = compute_mean_std(lppo_reward_by_bin_01)
lppo_bin_centers_001, lppo_mean_rewards_001, lppo_std_rewards_001 = compute_mean_std(lppo_reward_by_bin_001)
lppo_bin_centers_05, lppo_mean_rewards_05, lppo_std_rewards_05 = compute_mean_std(lppo_reward_by_bin_05)
lppo_bin_centers_075, lppo_mean_rewards_075, lppo_std_rewards_075 = compute_mean_std(lppo_reward_by_bin_075)
lppo_bin_centers_085, lppo_mean_rewards_085, lppo_std_rewards_085 = compute_mean_std(lppo_reward_by_bin_085)
lppo_bin_centers_1, lppo_mean_rewards_1, lppo_std_rewards_1 = compute_mean_std(lppo_reward_by_bin_1)
lppo_bin_centers_1_1, lppo_mean_rewards_1_1, lppo_std_rewards_1_1 = compute_mean_std(lppo_reward_by_bin_1_1)
lppo_bin_centers_1_5, lppo_mean_rewards_1_5, lppo_std_rewards_1_5 = compute_mean_std(lppo_reward_by_bin_1_5)
ly_bin_centers, ly_mean_rewards, ly_std_rewards = compute_mean_std(ly_reward_by_bin)
ppo_bin_centers, ppo_mean_rewards, ppo_std_rewards = compute_mean_std(ppo_reward_by_bin)


def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

lppo_mean_rewards = moving_average(lppo_mean_rewards, window_size=50)
lppo_std_rewards = moving_average(lppo_std_rewards, window_size=50)
lppo_mean_rewards_01 = moving_average(lppo_mean_rewards_01, window_size=50)
lppo_std_rewards_01 = moving_average(lppo_std_rewards_01, window_size=50)
lppo_mean_rewards_001 = moving_average(lppo_mean_rewards_001, window_size=50)
lppo_std_rewards_001 = moving_average(lppo_std_rewards_001, window_size=50)
lppo_mean_rewards_05 = moving_average(lppo_mean_rewards_05, window_size=50)
lppo_std_rewards_05 = moving_average(lppo_std_rewards_05, window_size=50)
lppo_mean_rewards_075 = moving_average(lppo_mean_rewards_075, window_size=50)
lppo_std_rewards_075 = moving_average(lppo_std_rewards_075, window_size=50)
lppo_mean_rewards_085 = moving_average(lppo_mean_rewards_085, window_size=50)
lppo_std_rewards_085 = moving_average(lppo_std_rewards_085, window_size=50)
lppo_mean_rewards_1 = moving_average(lppo_mean_rewards_1, window_size=50)
lppo_std_rewards_1 = moving_average(lppo_std_rewards_1, window_size=50)
lppo_mean_rewards_1_1 = moving_average(lppo_mean_rewards_1_1, window_size=50)
lppo_std_rewards_1_1 = moving_average(lppo_std_rewards_1_1, window_size=50)
lppo_mean_rewards_1_5 = moving_average(lppo_mean_rewards_1_5, window_size=50)
lppo_std_rewards_1_5 = moving_average(lppo_std_rewards_1_5, window_size=50)
ly_mean_rewards = moving_average(ly_mean_rewards, window_size=50)
ly_std_rewards = moving_average(ly_std_rewards, window_size=50)
ppo_mean_rewards = moving_average(ppo_mean_rewards, window_size=50)
ppo_std_rewards = moving_average(ppo_std_rewards, window_size=50)


# Plot the mean with shaded standard deviation for all algorithms
plt.figure(figsize=(10, 6))

# LPPO
# plt.plot(lppo_bin_centers, lppo_mean_rewards, label="$\mu=0.0$", color="blue")
# plt.fill_between(lppo_bin_centers, 
#                  np.array(lppo_mean_rewards) - np.array(lppo_std_rewards), 
#                  np.array(lppo_mean_rewards) + np.array(lppo_std_rewards), 
#                  color="blue", alpha=0.2)

# plt.plot(lppo_bin_centers_01, lppo_mean_rewards_01, label="$\mu=0.1$", color="green")
# plt.fill_between(lppo_bin_centers_01, 
#                  np.array(lppo_mean_rewards_01) - np.array(lppo_std_rewards_01), 
#                  np.array(lppo_mean_rewards_01) + np.array(lppo_std_rewards_01), 
#                  color="green", alpha=0.2)

# plt.plot(lppo_bin_centers_001, lppo_mean_rewards_001, label="$\mu=0.01$", color="red")
# plt.fill_between(lppo_bin_centers_001, 
#                  np.array(lppo_mean_rewards_001) - np.array(lppo_std_rewards_001), 
#                  np.array(lppo_mean_rewards_001) + np.array(lppo_std_rewards_001), 
#                  color="red", alpha=0.2)

# plt.plot(lppo_bin_centers_05, lppo_mean_rewards_05, label="$\mu=0.5$", color="orange")
# plt.fill_between(lppo_bin_centers_05, 
#                  np.array(lppo_mean_rewards_05) - np.array(lppo_std_rewards_05), 
#                  np.array(lppo_mean_rewards_05) + np.array(lppo_std_rewards_05), 
#                  color="orange", alpha=0.2)

plt.plot(lppo_bin_centers_075, lppo_mean_rewards_075, label="$\mu=0.75$", color="purple")
plt.fill_between(lppo_bin_centers_075,
                 np.array(lppo_mean_rewards_075) - np.array(lppo_std_rewards_075), 
                 np.array(lppo_mean_rewards_075) + np.array(lppo_std_rewards_075), 
                 color="purple", alpha=0.2)

# plt.plot(lppo_bin_centers_085, lppo_mean_rewards_085, label="$\mu=0.85$", color="brown")
# plt.fill_between(lppo_bin_centers_085,
#                     np.array(lppo_mean_rewards_085) - np.array(lppo_std_rewards_085),
#                     np.array(lppo_mean_rewards_085) + np.array(lppo_std_rewards_085),
#                     color="brown", alpha=0.2)

# plt.plot(lppo_bin_centers_1, lppo_mean_rewards_1, label="$\mu=1.0$", color="pink")
# plt.fill_between(lppo_bin_centers_1,
#                     np.array(lppo_mean_rewards_1) - np.array(lppo_std_rewards_1),
#                     np.array(lppo_mean_rewards_1) + np.array(lppo_std_rewards_1),
#                     color="pink", alpha=0.2)

# plt.plot(lppo_bin_centers_1_1, lppo_mean_rewards_1_1, label="$\mu=1.1$", color="gray")
# plt.fill_between(lppo_bin_centers_1_1,
#                     np.array(lppo_mean_rewards_1_1) - np.array(lppo_std_rewards_1_1),
#                     np.array(lppo_mean_rewards_1_1) + np.array(lppo_std_rewards_1_1),
#                     color="gray", alpha=0.2)

# plt.plot(lppo_bin_centers_1_5, lppo_mean_rewards_1_5, label="$\mu=1.5$", color="olive")
# plt.fill_between(lppo_bin_centers_1_5,
#                     np.array(lppo_mean_rewards_1_5) - np.array(lppo_std_rewards_1_5),
#                     np.array(lppo_mean_rewards_1_5) + np.array(lppo_std_rewards_1_5),
#                     color="olive", alpha=0.2)

plt.plot(ly_bin_centers, ly_mean_rewards, label="LY", color="cyan")
plt.fill_between(ly_bin_centers,
                    np.array(ly_mean_rewards) - np.array(ly_std_rewards),
                    np.array(ly_mean_rewards) + np.array(ly_std_rewards),
                    color="cyan", alpha=0.2)

plt.plot(ppo_bin_centers, ppo_mean_rewards, label="PPO", color="magenta")
plt.fill_between(ppo_bin_centers,
                    np.array(ppo_mean_rewards) - np.array(ppo_std_rewards),
                    np.array(ppo_mean_rewards) + np.array(ppo_std_rewards),
                    color="magenta", alpha=0.2)


# Add labels, title, and legend
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Quadrotor Rewards")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()