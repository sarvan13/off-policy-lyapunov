import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import argparse
import random

from algorithms.ly.ly import LYAgent
from algorithms.ppo.ppo import PPOAgent

from env.quad.quad_rotor_still import QuadStillEnv
from env.cartpole.cost_pend import CustomInvertedPendulumEnv
from env.bicycle.bicycle_model import KinematicBicycleEnv

parser = argparse.ArgumentParser(description='Train PPO/LY with command line arguments')
parser.add_argument('--N', type=int, default=1024, help='Update frequency')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--entropy_coeff', type=float, default=0.001, help='Entropy coefficient')
parser.add_argument('--n_steps', type=int, default=1_000_000, help='Number of steps')
parser.add_argument('--modelType', type=str, default="ly", help='Model type: ppo or ly')
parser.add_argument('--env', type=str, default="Bicycle-v1", help='Environment name')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--torch_deterministic', type=bool, default=True, help='Use deterministic mode for PyTorch')
args = parser.parse_args()

update_freq = args.N
batch_size = args.batch_size
n_epochs = args.n_epochs
entropy_coeff = args.entropy_coeff
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

if modelType == "ppo":
    agent = PPOAgent(input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0], max_action=env.action_space.high, 
                     batch_size=batch_size, gamma=0.9, update_freq=update_freq, n_epochs=n_epochs, entropy_coeff=entropy_coeff, save_dir=data_path)
elif modelType == "ly":
    agent = LYAgent(input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0], max_action=env.action_space.high, 
                        dt=env.unwrapped.dt, equilibrium_state=equilibrium_state, entropy_coeff=entropy_coeff,
                        batch_size=batch_size, gamma=0.9, update_freq=update_freq, n_epochs=n_epochs, save_dir=data_path)    
else:
    raise ValueError("Invalid model type")

agent.save_models()

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
        action, probs, val = agent.choose_action(state)
        next_state, cost, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, probs, val, cost, next_state, done)

        state = next_state

        episode_cost += cost
        episode_steps += 1
        global_steps += 1

        if global_steps % update_freq == 0 and global_steps > 0:
            agent.calculate_advantages()
            if modelType == "ly":
                agent.train_lyapunov()
            agent.learn()

        if terminated or truncated:
            break

    episode_num += 1
    reward_arr.append(episode_cost)
    avg_reward = np.mean(reward_arr[-100:])
    step_arr.append(global_steps)
    

    if avg_reward > best_reward:
        best_reward = avg_reward
        agent.save_models()
        print(f"Best model saved at episode {episode_num} with average reward {avg_reward}")

    if episode_num % 50 == 0:
        print(f"Episode {episode_num} - Cost: {episode_cost}, Average Cost: {avg_reward}, Steps: {episode_steps}, Global Steps: {global_steps}")

    state, _ = env.reset()
    done = False
    

    # print(f"Episode {k} - Cost: {episode_cost}, Steps: {episode_steps}")
    # if modelType == "lsac":
    #     print(f"Beta: {agent.beta.item()}")

env.close()

np.save(os.path.join(data_path, "reward_arr.npy"), np.array(reward_arr))
np.save(os.path.join(data_path, "step_arr.npy"), np.array(step_arr))

# agent.save()

