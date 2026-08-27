import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mu_list = [0.05, 0.001, 0.005, 0.0005, 0.0001]
mu_list = sorted(mu_list, reverse=False)
# mu_list = [0.001]
# Plot the mean with shaded standard deviation for all algorithms
plt.figure(figsize=(10, 6))

for mu in mu_list:
    # Initialize lists for rewards and steps
    rew, steps = [], []

    # Load rewards and step arrays for 10 seeds for each algorithm
    for i in range(1, 11):
        rew.append(np.load(os.path.join(curr_dir, "data", "reg", "Pendulum-v1", "beta1", "mu_"+str(mu),"seed_" + str(i+10), "reward_arr.npy")))
        episode_steps= np.load(os.path.join(curr_dir, "data", "reg", "Pendulum-v1", "beta1", "mu_"+str(mu),"seed_" + str(i+10), "step_arr.npy"))
        steps.append(np.cumsum(episode_steps))

    mean = np.mean(rew, axis=0)
    std = np.std(rew, axis=0)

    def moving_average(data, window_size=50):
        return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 
    
    mean_rewards = moving_average(mean, window_size=50)
    std_rewards = moving_average(std, window_size=50)

    x = steps[0]

    # LPPO
    plt.plot(x, mean_rewards, label="$\mu$ = "+str(mu))
    plt.fill_between(x, 
                    np.array(mean_rewards) - np.array(std_rewards), 
                    np.array(mean_rewards) + np.array(std_rewards), 
                    alpha=0.2)

# Add labels, title, and legend
plt.xlim(0, 0.1e6)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Pendulum Regularized")
plt.legend(loc="lower right")
plt.grid(True)

# Show the plot
plt.show()