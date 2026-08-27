import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

struct_dir = r"C:\Users\Sarvan\Desktop\School\UVIC\thesis\off-policy-lyapunov\data\struct\Pendulum-v1\beta_0.5\mu_0.001"
desc_dir = r"C:\Users\Sarvan\Desktop\School\UVIC\thesis\off-policy-lyapunov\data\desc\Pendulum-v1\mu_0.1"
zero_dir = r"C:\Users\Sarvan\Desktop\School\UVIC\thesis\off-policy-lyapunov\data\desc\Pendulum-v1\mu_0.0"
reg_dir = r"C:\Users\Sarvan\Desktop\School\UVIC\thesis\off-policy-lyapunov\data\reg\Pendulum-v1\beta1\mu_0.0005"
sac_dir = r"C:\Users\Sarvan\Desktop\School\UVIC\off-policy-lyapunov\data\Pendulum-v1\sac"
ly_dir = r"C:\Users\Sarvan\Desktop\School\UVIC\off-policy-lyapunov\data\Pendulum-v1\ly"
ppo_dir = r"C:\Users\Sarvan\Desktop\School\UVIC\off-policy-lyapunov\data\Pendulum-v1\ppo"
lac_dir = r"C:\Users\Sarvan\Desktop\School\UVIC\off-policy-lyapunov\data\Pendulum-v1\lac"

dir_list = [struct_dir, desc_dir, reg_dir, zero_dir, sac_dir, ly_dir, ppo_dir, lac_dir]
dir_labels = ["LSAC Structural Changes", "LSAC Min Descent", "LSAC Regularized", "LSAC No Scale", "SAC", "LY", "PPO", "LAC"]
plt.figure(figsize=(10, 6))

for dir, label in zip(dir_list, dir_labels):
    # Initialize lists for rewards and steps
    rew, steps = [], []

    # Load rewards and step arrays for 10 seeds for each algorithm
    for i in range(1, 11):
        if label == "SAC" or label == "PPO" or label == "LAC" or label == "LY":
            rew.append(np.load(os.path.join(dir, "seed_" + str(i), "reward_arr.npy")))
            episode_steps= np.load(os.path.join(dir, "seed_" + str(i), "step_arr.npy"))
            if label == "LY" or label == "PPO":
                steps.append(episode_steps)
            else:
                steps.append(np.cumsum(episode_steps))
        else:
            rew.append(np.load(os.path.join(dir, "seed_" + str(i+10), "reward_arr.npy")))
            episode_steps= np.load(os.path.join(dir, "seed_" + str(i+10), "step_arr.npy"))
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
    plt.plot(steps[0], rew0_mean, label=label)
    plt.fill_between(steps[0], np.array(rew0_mean) - np.array(rew0_std), np.array(rew0_mean) + np.array(rew0_std), alpha=0.2)

# Add labels, title, and legend
plt.xlim(0, 0.1e6)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Pendulum Reward Comparison")
plt.legend(loc="lower right")
plt.grid(True)

# Show the plot
plt.show()