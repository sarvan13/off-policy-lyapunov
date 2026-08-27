import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mu_list = [0.0, 0.1, 0.01, 0.5, 0.15, 0.25, 0.75]
# Plot the mean with shaded standard deviation for all algorithms
plt.figure(figsize=(10, 6))

for mu in mu_list:
    # Initialize lists for rewards and steps
    rew, steps = [], []

    # Load rewards and step arrays for 10 seeds for each algorithm
    for i in range(1, 11):
        rew.append(np.load(os.path.join(curr_dir, "data", "desc", "Pendulum-v1", "mu_"+str(mu), "seed_" + str(i+10), "reward_arr.npy")))
        episode_steps= np.load(os.path.join(curr_dir, "data", "desc", "Pendulum-v1", "mu_"+str(mu), "seed_" + str(i+10), "step_arr.npy"))
        steps.append(np.cumsum(episode_steps))

    rew0_mean = np.mean(rew, axis=0)
    rew0_std = np.std(rew, axis=0)

    # Function to compute the moving average
    def moving_average(data, window_size=50):
        return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

    # Compute the moving averages
    rew0_mean = moving_average(rew0_mean, window_size=50)
    rew0_std = moving_average(rew0_std, window_size=50)


    # SAC plot with shaded std
    plt.plot(steps[0], rew0_mean, label="$\mu$ = "+str(mu))
    plt.fill_between(steps[0], np.array(rew0_mean) - np.array(rew0_std), np.array(rew0_mean) + np.array(rew0_std), alpha=0.2)


# Add labels, title, and legen
plt.xlim(0, 0.1e6)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Pendulum Min Descent")
plt.legend(loc="lower right")
plt.grid(True)

# Show the plot
plt.show()