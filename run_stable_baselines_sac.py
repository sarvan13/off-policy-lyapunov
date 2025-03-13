import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym
from env.quad import QuadRateEnv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import multiprocessing

# def make_env():
#     return gym.make("Quadrotor-v1")  # Replace with your environment

# # Create multiple environments
# n_envs = multiprocessing.cpu_count() // 2  # Use half of CPU cores
# envs = SubprocVecEnv([lambda: make_env() for _ in range(n_envs)])

# Create the environment
env = gym.make("Quadrotor-v1")

# Define a callback to log rewards and steps
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.steps = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate rewards for the current episode
        self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if the episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.steps.append(self.num_timesteps)
            self.current_episode_reward = 0  # Reset for the next episode
        
        return True

# Instantiate the callback
reward_callback = RewardCallback()

# Instantiate the PPO model
model = SAC('MlpPolicy', env, batch_size=256, verbose=1)

# Train the model
num_episodes = 1000
num_steps = 5_000_000
model.learn(total_timesteps=num_steps, callback=reward_callback)

# Close the environment
env.close()

# Plot the rewards
plt.figure(figsize=(10, 5))
plt.plot(reward_callback.steps, reward_callback.episode_rewards)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Rewards for SAC on Quadrotor-v1")
plt.grid(True)

# Save the plot to a file
plt.savefig('sac_training_rewards.png')

# Save the rewards and steps to a file
np.savez('sac_training_data.npz', rewards=reward_callback.episode_rewards, steps=reward_callback.steps)
model.save("sac_quadrotor")