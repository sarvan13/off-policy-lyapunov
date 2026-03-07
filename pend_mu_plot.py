import matplotlib.pyplot as plt
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

rew0 = []
rew01 = []
rew015 = []
rew025 = []
rew05 = []
rew075 = []

for i in range(1,11):
    rew0.append(np.load(os.path.join(curr_dir, "Pendulum-v1", "lsac", "mu_0.0", "seed_" + str(i+10), "reward_arr.npy")))
    rew01.append(np.load(os.path.join(curr_dir, "Pendulum-v1", "lsac", "mu_0.1", "seed_" + str(i+10), "reward_arr.npy")))
    rew015.append(np.load(os.path.join(curr_dir, "Pendulum-v1", "lsac", "mu_0.15", "seed_" + str(i+10), "reward_arr.npy")))
    rew025.append(np.load(os.path.join(curr_dir, "Pendulum-v1", "lsac", "mu_0.25", "seed_" + str(i+10), "reward_arr.npy")))
    rew05.append(np.load(os.path.join(curr_dir, "Pendulum-v1", "lsac", "mu_0.5", "seed_" + str(i+10), "reward_arr.npy")))
    rew075.append(np.load(os.path.join(curr_dir, "Pendulum-v1", "lsac", "mu_0.75", "seed_" + str(i+10), "reward_arr.npy")))

rew0_mean = np.mean(rew0, axis=0)
rew0_std = np.std(rew0, axis=0)
rew01_mean = np.mean(rew01, axis=0)
rew01_std = np.std(rew01, axis=0)
rew015_mean = np.mean(rew015, axis=0)
rew015_std = np.std(rew015, axis=0)
rew025_mean = np.mean(rew025, axis=0)
rew025_std = np.std(rew025, axis=0)
rew05_mean = np.mean(rew05, axis=0)
rew05_std = np.std(rew05, axis=0)
rew075_mean = np.mean(rew075, axis=0)
rew075_std = np.std(rew075, axis=0)
# Function to compute the moving average
def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

# Compute the moving averages
rew0_mean = moving_average(rew0_mean, window_size=50)
rew0_std = moving_average(rew0_std, window_size=50)
rew01_mean = moving_average(rew01_mean, window_size=50)
rew01_std = moving_average(rew01_std, window_size=50)
rew015_mean = moving_average(rew015_mean, window_size=50)
rew015_std = moving_average(rew015_std, window_size=50)
rew025_mean = moving_average(rew025_mean, window_size=50)
rew025_std = moving_average(rew025_std, window_size=50)
rew05_mean = moving_average(rew05_mean, window_size=50)
rew05_std = moving_average(rew05_std, window_size=50)
rew075_mean = moving_average(rew075_mean, window_size=50)
rew075_std = moving_average(rew075_std, window_size=50)


# Plot the moving averages with shaded standard deviation
plt.figure(figsize=(10, 6))
x = np.arange(len(rew0_mean))  # X-axis values

# SAC plot with shaded std
plt.plot(x, rew0_mean, label="$\mu=0$", color="blue")
plt.fill_between(x, np.array(rew0_mean) - np.array(rew0_std), np.array(rew0_mean) + np.array(rew0_std), color="blue", alpha=0.2)
# LSAC plot with shaded std
plt.plot(x, rew01_mean, label="$\mu=0.1$", color="green")
plt.fill_between(x, np.array(rew01_mean) - np.array(rew01_std), np.array(rew01_mean) + np.array(rew01_std), color="green", alpha=0.2)

plt.plot(x, rew015_mean, label="$\mu=0.15$", color="orange")
plt.fill_between(x, np.array(rew015_mean) - np.array(rew015_std), np.array(rew015_mean) + np.array(rew015_std), color="orange", alpha=0.2)

plt.plot(x, rew025_mean, label="$\mu=0.25$", color="purple")
plt.fill_between(x, np.array(rew025_mean) - np.array(rew025_std), np.array(rew025_mean) + np.array(rew025_std), color="purple", alpha=0.2)

plt.plot(x, rew05_mean, label="$\mu=0.5$", color="brown")
plt.fill_between(x, np.array(rew05_mean) - np.array(rew05_std), np.array(rew05_mean) + np.array(rew05_std), color="brown", alpha=0.2)

plt.plot(x, rew075_mean, label="$\mu=0.75$", color="red")
plt.fill_between(x, np.array(rew075_mean) - np.array(rew075_std), np.array(rew075_mean) + np.array(rew075_std), color="red", alpha=0.2)


# Add labels, title, and legend
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Average Rewards (Last 50 Episodes)")
plt.title("LSAC Reward on Pendulum-v1 with Different $\mu$ Values")


# Show the plot
plt.show()