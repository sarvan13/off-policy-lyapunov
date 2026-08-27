import matplotlib.pyplot as plt
import numpy as np
import os

curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize lists for rewards and steps
rew, steps = [], []


# Load rewards and step arrays for 10 seeds for each algorithm
for i in range(1, 11):
    rew.append(np.load(os.path.join(curr_dir, "data", "struct", "Pendulum-v1", "beta_1", "seed_" + str(i+10), "reward_arr.npy")))
    episode_steps= np.load(os.path.join(curr_dir, "data", "struct", "Pendulum-v1", "beta_1", "seed_" + str(i+10), "step_arr.npy"))
    steps.append(np.cumsum(episode_steps))

rew0_mean = np.mean(rew, axis=0)
rew0_std = np.std(rew, axis=0)

# Function to compute the moving average
def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

# Compute the moving averages
rew0_mean = moving_average(rew0_mean, window_size=50)
rew0_std = moving_average(rew0_std, window_size=50)


# Plot the moving averages with shaded standard deviation
plt.figure(figsize=(10, 6))
x = np.arange(len(rew0_mean))  # X-axis values
x = steps[0]

# SAC plot with shaded std
plt.plot(x[:500], rew0_mean[:500], color="blue")
plt.fill_between(x[:500], np.array(rew0_mean[:500]) - np.array(rew0_std[:500]), np.array(rew0_mean[:500]) + np.array(rew0_std[:500]), color="blue", alpha=0.2)

# Add labels, title, and legend
plt.xlim(0, 0.1e6)
plt.xlabel("Steps")
plt.ylabel("Average Rewards (Last 50 Episodes)")
plt.title("LSAC Pendulum-v1 with Structural Changes NN")

# Show the plot
plt.show()