import gymnasium as gym
import numpy as np
from algorithms.lsac.agent import LSACAgent
from algorithms.ly.ly import LYAgent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random
import torch
import os

# from clean_ppo import Agent, make_env
# from clean_lppo import Agent
# env_name = "Hopper-v4"
# env_name = "Quadrotor-Still-v1"
env_name = "Pendulum-v1"

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

env = gym.make(env_name, render_mode="human")
equilibrium_state = torch.tensor([np.array([np.cos(0), np.sin(0), 0])], dtype=torch.float)

# agent = LSACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, 
                        #   dt=env.unwrapped.dt, equilibrium_state=equilibrium_state, save_dir="data")

agent = LYAgent(n_actions=env.action_space.shape[0], batch_size=64,
                    alpha=0.0003, n_epochs=10, dt=env.unwrapped.dt,
                    input_dims=env.observation_space.shape[0],
                    max_action=env.action_space.high, update_freq=2048,
                    equilibrium_state=equilibrium_state, entropy_coeff=0.001)

# curr_dir = os.getcwd()
# agent_path = os.path.join(curr_dir, "runs", "Hopper-v4__clean_ppo__1__1743787519", "clean_ppo.cleanrl_model.pth")

agent.load_models()

obs, info = env.reset(seed=2)
ly_eq_data = []
epsilon = 0.1
record_data = False


for i in range(1):
    total_reward = 0
    done = False
    while not done:
        action, _, _ = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        theta = np.atan2(obs[1], obs[0])

        if np.abs(theta) < epsilon:
            record_data = True
        if record_data:
            ly_eq_data.append(theta)
        # ly_eq_data.append(theta)

        total_reward += reward
        done = (terminated | truncated)

        obs = next_obs

env.close()
print(total_reward)

env = gym.make(env_name, render_mode="human")
agent = LSACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, 
                          dt=env.unwrapped.dt, equilibrium_state=equilibrium_state, save_dir="data")

agent.load()

obs, info = env.reset(seed=2)
lsac_eq_data = []
epsilon = 0.1
record_data = False

for i in range(1):
    total_reward = 0
    done = False
    while not done:
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        theta = np.atan2(obs[1], obs[0])

        if np.abs(theta) < epsilon:
            record_data = True
        if record_data:
            lsac_eq_data.append(theta)

        # lsac_eq_data.append(theta)

        total_reward += reward
        done = (terminated | truncated)


        obs = next_obs

env.close()
print(total_reward)

plt.plot(ly_eq_data)
plt.plot(lsac_eq_data)
plt.show()
