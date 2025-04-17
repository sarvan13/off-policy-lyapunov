import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import os
import random
import numpy as np
import os

from algorithms.sac.agent import SACAgent
from algorithms.lsac.agent import LSACAgent
from algorithms.lac.agent import LAC
from algorithms.ly.ly import LYAgent
from algorithms.ppo.ppo import PPOAgent

from env.cartpole.cost_pend import CustomInvertedPendulumEnv

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

env_name = "CustomCart-v1"

curr_dir = os.path.dirname(os.path.abspath(__file__))
lsac_dir = os.path.join(curr_dir, "data", env_name, "lsac", "seed_32")
sac_dir = os.path.join(curr_dir, "data", env_name, "sac", "seed_1")
ly_dir = os.path.join(curr_dir, "data", env_name, "ly", "seed_1")

env = gym.make(env_name)
# agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, 
#                 save_dir=sac_dir)

equilibrium_state = torch.zeros((1, env.observation_space.shape[0]), dtype=torch.float)
# equilibrium_state[0][0] = 1.0
agent = LSACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, 
                          dt=env.unwrapped.dt, equilibrium_state=equilibrium_state, save_dir=lsac_dir)

# agent = LYAgent(input_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0], max_action=env.action_space.high, 
                        # dt=env.unwrapped.dt, update_freq=2048, equilibrium_state=equilibrium_state, save_dir=ly_dir)

agent.load()
# agent.load_models()

obs, info = env.reset(seed=7)
lsac_x = []
lsac_lyapunov = []

for i in range(1):
    total_reward = 0
    done = False
    while not done:
        action = agent.choose_action(obs)
        # action, probs, val = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        l = agent.lyapunov(torch.tensor([obs], dtype=torch.float).to(agent.actor.device), torch.tensor([action], dtype=torch.float).to(agent.actor.device))
        # l = agent.lyapunov(torch.tensor([obs], dtype=torch.float).to(agent.actor.device))
        lsac_lyapunov.append(l.cpu().detach().numpy()[0][0])
        x = obs[0]
        lsac_x.append(x)

        # lsac_eq_data.append(theta)

        total_reward += reward
        done = (terminated | truncated)


        obs = next_obs

lsac_rew = total_reward

equilibrium_state = equilibrium_state.to(agent.actor.device)
eq_action, _ = agent.actor.forward(equilibrium_state)
eq_lyapunov = agent.lyapunov(equilibrium_state, eq_action).cpu().detach().numpy()[0][0]
# eq_lyapunov = agent.lyapunov(equilibrium_state).cpu().detach().numpy()[0][0]


print(f"Equilibrium Lyapunov: {eq_lyapunov}")

plt.plot(lsac_x, label="LSAC")
plt.xlabel("Steps")
plt.ylabel("X Position")
plt.title("X Position vs Steps")
plt.legend()
plt.show()

plt.plot(lsac_lyapunov, label="LSAC")
plt.xlabel("Steps")
plt.ylabel("Lyapunov Value")
plt.title("Lyapunov Value vs Steps")
plt.legend()
plt.show()

beta = np.load(lsac_dir + "\\beta_arr.npy")
plt.plot(beta, label="LSAC")
plt.xlabel("Steps")
plt.ylabel("Beta")
plt.title("Beta vs Steps")
plt.legend()
plt.show()

env.close()