import gymnasium as gym
import numpy as np
from algorithms.ly.ly import LYAgent
import matplotlib.pyplot as plt
import torch as T
from env.quad import QuadRateEnv

env = gym.make('Quadrotor-v1', render_mode='human')
N = 32*2048
batch_size = 256
n_epochs = 10
alpha = 0.0003
agent = LYAgent(n_actions=env.action_space.shape[0], batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, dt=env.unwrapped.dt,
                    input_dims=env.observation_space.shape[0],
                    max_action=env.action_space.high, entropy_coeff=0.001)
agent.load_models()
# n_games = 2
# theta_arr = []
# theta_dot_arr = []
# 
# for i in range(n_games):
#     print('Game:', i)
#     observation, _ = env.reset()
#     done = False
#     while not done:
#         action, prob, val = agent.choose_action(observation)
#         observation_, reward, terminated, truncated, info = env.step(action)
#         cos_theta, sin_theta, theta_dot = observation_
#         theta_arr.append(np.arctan2(sin_theta, cos_theta))
#         theta_dot_arr.append(theta_dot)
#         done = terminated or truncated
#         observation = observation_

# Parameters
num_episodes = 1  # Number of episodes to simulate
steps_per_episode = 1000  # Number of steps per episode
lyapunov_arr = []

# Function to run an episode and collect data
def run_episode():
    total_cost = 0
    obs, _ = env.reset()
    theta_list = []
    theta_dot_list = []
    for _ in range(steps_per_episode):
        action, prob, val = agent.choose_action(obs)
        obs, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # with T.no_grad():
        #     lyapunov_val = agent.lyapunov(T.tensor([-obs], dtype=T.float).to(agent.actor.device), T.tensor([action], dtype=T.float).to(agent.actor.device))
        #     lyapunov_arr.append(lyapunov_val.item())
    
        total_cost += cost
        if done:
            break
    
    print(total_cost)
    return theta_list, theta_dot_list

# Run multiple episodes and collect trajectories
# trajectories = [run_episode() for _ in range(num_episodes)]

run_episode()

env.close()
# lyapunov_arr_avg = [np.mean(lyapunov_arr[max(0, i-5):i+1]) for i in range(len(lyapunov_arr))]
# plt.plot(lyapunov_arr_avg)
# plt.xlabel("Timesteps")
# plt.ylabel("Lyapunov Value")
# plt.title("Lyapunov Value over Time")
# plt.grid(True)
# plt.show()

# # Plot trajectories
# plt.figure(figsize=(10, 5))
# for i, (theta_list, theta_dot_list) in enumerate(trajectories):
#     plt.plot(theta_list, theta_dot_list, label=f'Episode {i+1}')
# plt.xlabel("Theta (rad)")
# plt.ylabel("Theta dot (rad/s)")
# plt.title("Phase Portraits of Pendulum for LSAC")
# plt.legend()
# plt.grid(True)
# plt.show()
