import matplotlib.pyplot as plt
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

sac_rew = []
lsac_rew = []
ly_rew = []
ppo_rew = []
lac_rew = []

for i in range(1,11):
    sac_rew.append(np.load(os.path.join(curr_dir, "data", "Pendulum-v1", "sac", "seed_" + str(i), "reward_arr.npy")))
    lsac_rew.append(np.load(os.path.join(curr_dir, "data", "Pendulum-v1", "lsac", "seed_" + str(i+10), "reward_arr.npy")))
    ly_rew.append(np.load(os.path.join(curr_dir, "data", "Pendulum-v1", "ly", "seed_" + str(i), "reward_arr.npy")))
    ppo_rew.append(np.load(os.path.join(curr_dir, "data", "Pendulum-v1", "ppo", "seed_" + str(i), "reward_arr.npy")))
    lac_rew.append(np.load(os.path.join(curr_dir, "data", "Pendulum-v1", "lac", "seed_" + str(i), "reward_arr.npy")))

sac_mean = np.mean(sac_rew, axis=0)
sac_std = np.std(sac_rew, axis=0)
lsac_mean = np.mean(lsac_rew, axis=0)
lsac_std = np.std(lsac_rew, axis=0)
ly_mean = np.mean(ly_rew, axis=0)
ly_std = np.std(ly_rew, axis=0)
ppo_mean = np.mean(ppo_rew, axis=0)
ppo_std = np.std(ppo_rew, axis=0)
lac_mean = np.mean(lac_rew, axis=0)
lac_std = np.std(lac_rew, axis=0)


# Function to compute the moving average
def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

# Compute the moving averages
sac_mean = moving_average(sac_mean, window_size=50)
sac_std = moving_average(sac_std, window_size=50)
lsac_mean = moving_average(lsac_mean, window_size=50)
lsac_std = moving_average(lsac_std, window_size=50)
ly_mean = moving_average(ly_mean, window_size=50)
ly_std = moving_average(ly_std, window_size=50)
ppo_mean = moving_average(ppo_mean, window_size=50)
ppo_std = moving_average(ppo_std, window_size=50)
lac_mean = moving_average(lac_mean, window_size=50)
lac_std = moving_average(lac_std, window_size=50)


# Plot the moving averages with shaded standard deviation
plt.figure(figsize=(10, 6))
x = np.arange(len(sac_mean))  # X-axis values

# SAC plot with shaded std
plt.plot(x, sac_mean, label="SAC", color="blue")
plt.fill_between(x, np.array(sac_mean) - np.array(sac_std), np.array(sac_mean) + np.array(sac_std), color="blue", alpha=0.2)

# LSAC plot with shaded std
plt.plot(x, lsac_mean, label="LSAC", color="green")
plt.fill_between(x, np.array(lsac_mean) - np.array(lsac_std), np.array(lsac_mean) + np.array(lsac_std), color="green", alpha=0.2)

plt.plot(x, ly_mean, label="LY", color="orange")
plt.fill_between(x, np.array(ly_mean) - np.array(ly_std), np.array(ly_mean) + np.array(ly_std), color="orange", alpha=0.2)

plt.plot(x, ppo_mean, label="PPO", color="red")
plt.fill_between(x, np.array(ppo_mean) - np.array(ppo_std), np.array(ppo_mean) + np.array(ppo_std), color="red", alpha=0.2)

plt.plot(x, lac_mean, label="LAC", color="purple")
plt.fill_between(x, np.array(lac_mean) - np.array(lac_std), np.array(lac_mean) + np.array(lac_std), color="purple", alpha=0.2)


# Add labels, title, and legend
plt.legend()

# Show the plot
plt.show()