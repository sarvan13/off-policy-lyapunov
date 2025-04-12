import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import argparse
import random

from algorithms.sac.agent import SACAgent
from algorithms.lsac.agent import LSACAgent

from env.quad.quad_rotor_still import QuadStillEnv
from env.cartpole.cost_pend import CustomInvertedPendulumEnv
from env.bicycle.bicycle_model import KinematicBicycleEnv

parser = argparse.ArgumentParser(description='Train SAC/LSAC with command line arguments')
# parser.add_argument('--N', type=int, default=2048, help='Update frequency')
parser.add_argument('--n_steps', type=int, default=200_000, help='Number of steps')
parser.add_argument('--modelType', type=str, default="sac", help='Model type: sac or lsac')
parser.add_argument('--env', type=str, default="Pendulum-v1", help='Environment name')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--torch_deterministic', type=bool, default=True, help='Use deterministic mode for PyTorch')
args = parser.parse_args()

total_steps = args.n_steps
modelType = args.modelType
env_name = args.env

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

# Create an instance of the custom environment
env = gym.make(env_name)

if env_name == "Pendulum-v1":
    equilibrium_state = torch.tensor([np.array([np.cos(0), np.sin(0), 0])], dtype=torch.float)
else:
    equilibrium_state = torch.zeros((1, env.observation_space.shape[0]), dtype=torch.float)

curr_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_dir, "data", env_name, modelType, "seed_" + str(args.seed))
os.makedirs(data_path, exist_ok=True)

if modelType == "sac":
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, save_dir=data_path)
elif modelType == "lsac":
    agent = LSACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, 
                        dt=env.unwrapped.dt, equilibrium_state=equilibrium_state, save_dir=data_path)    
else:
    raise ValueError("Invalid model type")

agent.save()

# Reset the environment to get the initial state
state, info = env.reset(seed=args.seed)
done = False

reward_arr = []
step_arr = []
best_reward = -np.inf
global_steps = 0
episode_num = 0

while global_steps < total_steps:
    episode_cost = 0
    episode_steps = 0
    while not done:
        action = agent.choose_action(state, reparameterize=False)
        next_state, cost, terminated, truncated, _ = env.step(action)

        agent.remember((state, action, cost, next_state, terminated))

        state = next_state

        episode_cost += cost
        episode_steps += 1

        if terminated or truncated:
            break

    episode_num += 1
    reward_arr.append(episode_cost)
    avg_reward = np.mean(reward_arr[-100:])
    step_arr.append(episode_steps)
    global_steps += episode_steps

    if avg_reward > best_reward:
        best_reward = avg_reward
        agent.save()
        print(f"Best model saved at episode {episode_num} with average reward {avg_reward}")

    if episode_num % 50 == 0:
        print(f"Episode {episode_num} - Cost: {episode_cost}, Average Cost: {avg_reward}, Steps: {episode_steps}, Global Steps: {global_steps}")

    state, _ = env.reset()
    
    for j in range(episode_steps):
        if modelType == "lsac":
            agent.learn_lyapunov()
        agent.train()

    # print(f"Episode {k} - Cost: {episode_cost}, Steps: {episode_steps}")
    # if modelType == "lsac":
    #     print(f"Beta: {agent.beta.item()}")

env.close()

np.save(os.path.join(data_path, "reward_arr.npy"), np.array(reward_arr))
np.save(os.path.join(data_path, "step_arr.npy"), np.array(step_arr))

# agent.save()

