import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import pathlib
import argparse
import random

from algorithms.sac.agent import SACAgent
from algorithms.lsac.agent import LSACAgent
from algorithms.lac.agent import LAC

from env.quad.quad_rotor_still import QuadStillEnv
from env.cartpole.cost_pend import CustomInvertedPendulumEnv
from env.bicycle.bicycle_model import KinematicBicycleEnv

parser = argparse.ArgumentParser(description='Train SAC/LSAC with command line arguments')
# parser.add_argument('--N', type=int, default=2048, help='Update frequency')
parser.add_argument('--n_steps', type=int, default=1_000_000, help='Number of steps')
parser.add_argument('--mu', type=float, default=0.1, help='Lyapunov regularization parameter')
parser.add_argument('--modelType', type=str, default="sac", help='Model type: sac or lsac')
parser.add_argument('--env', type=str, default="Bicycle-v1", help='Environment name')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--torch_deterministic', type=bool, default=True, help='Use deterministic mode for PyTorch')
args = parser.parse_args()

mu = args.mu
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

# 1. Resolve the REAL physical path, bypassing the /home/... symlink
# This ensures consistency between what you see in 'pwd' and what the OS sees
script_dir = pathlib.Path(__file__).parent.resolve()

# 2. Construct the path: project/data/env/model/mu_value/seed_X
# Using f"mu_{args.mu}" creates a clean folder name like "mu_0.01"
data_path = (
    script_dir / 
    "data" / 
    env_name / 
    modelType / 
    f"mu_{args.mu}" / 
    f"seed_{args.seed}"
)

# 3. Create the directory
# parents=True ensures /data, /Pendulum-v1, etc. are all created if missing
data_path.mkdir(parents=True, exist_ok=True)

# 4. CRITICAL: Use the string version of the path for your logger/saving function
final_path_str = str(data_path.absolute())

if modelType == "sac":
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, save_dir=data_path, gamma=0.9)
elif modelType == "lsac":
    agent = LSACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, 
                        dt=env.unwrapped.dt, equilibrium_state=equilibrium_state, save_dir=data_path, gamma=0.9, mu=mu)    
elif modelType == "lac":
    agent = LAC(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, alpha=0.1, save_dir=data_path, gamma=0.9)
else:
    raise ValueError("Invalid model type")

agent.save()

# Reset the environment to get the initial state
state, info = env.reset(seed=args.seed)
done = False

reward_arr = []
step_arr = []
beta_arr = []
ly_loss = []
best_reward = -np.inf
global_steps = 0
episode_num = 0

while global_steps < total_steps:
    episode_cost = 0
    episode_steps = 0
    while not done:
        action = agent.choose_action(state, reparameterize=False)
        next_state, cost, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        if modelType == "lac":
            agent.remember((state, action, -cost, next_state, done))
        else:
            agent.remember((state, action, cost, next_state, done))

        state = next_state

        episode_cost += cost
        episode_steps += 1


    episode_num += 1
    reward_arr.append(episode_cost)
    avg_reward = np.mean(reward_arr[-100:])
    step_arr.append(episode_steps)
    global_steps += episode_steps

    if avg_reward > best_reward:
        best_reward = avg_reward
        agent.save()
        print(f"Best model saved at {data_path}: episode {episode_num} with average reward {avg_reward}")

    if episode_num % 50 == 0:
        print(f"Episode {episode_num} - Cost: {episode_cost}, Average Cost: {avg_reward}, Steps: {episode_steps}, Avg Steps: {np.mean([step_arr[-100:]])}, Global Steps: {global_steps}")

    state, _ = env.reset()
    done = False
    
    for j in range(episode_steps):
        if modelType == "lsac":
            loss = agent.learn_lyapunov()
            if j == episode_steps - 1 and loss is not None:
                ly_loss.append(loss.item())
        agent.train()

    # print(f"Episode {k} - Cost: {episode_cost}, Steps: {episode_steps}")
    if modelType == "lsac":
        beta_arr.append(agent.beta.item())

env.close()

np.save(os.path.join(data_path, "reward_arr.npy"), np.array(reward_arr))
np.save(os.path.join(data_path, "step_arr.npy"), np.array(step_arr))
np.save(os.path.join(data_path, "beta_arr.npy"), np.array(beta_arr))
np.save(os.path.join(data_path, "ly_loss.npy"), np.array(ly_loss))

# agent.save()

