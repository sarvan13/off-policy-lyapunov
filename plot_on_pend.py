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
    ppo_rew.append(np.load(os.path.join(curr_dir, "data", "Pendulum-v1", "ly", "seed_" + str(i), "reward_arr.npy")))


# Function to compute the moving average
def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 



for i in range(len(ppo_rew)):
    plt.plot(moving_average(ppo_rew[i], window_size=50), label=f"Seed {i+1}")
    

# Add labels, title, and legend
plt.legend()

# Show the plot
plt.show()