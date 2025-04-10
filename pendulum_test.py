import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

from algorithms.sac.agent import SACAgent
from algorithms.lsac.agent import LSACAgent

from env.quad.quad_rotor_still import QuadStillEnv
from env.cartpole.cost_pend import CustomInvertedPendulumEnv
from env.bicycle.bicycle_model import KinematicBicycleEnv

def train_inverted_pendulum(modelType):
    # Create an instance of the custom environment
    # env = gym.make("Bicycle-v1")
    # equilibrium_state = torch.zeros((1, env.observation_space.shape[0]), dtype=torch.float)
    env = gym.make("Pendulum-v1")
    equilibrium_state = torch.tensor([np.array([np.cos(0), np.sin(0), 0])], dtype=torch.float)

    # Reset the environment to get the initial state
    state, info = env.reset()
    done = False
    total_reward = 0

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_dir, "data")
    
    if modelType == "sac":
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high)
    elif modelType == "lsac":
        agent = LSACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, 
                          dt=env.unwrapped.dt, equilibrium_state=equilibrium_state, save_dir="data")    
    else:
        raise ValueError("Invalid model type")
    agent.save()

    reward_arr = []
    max_num_episodes = 1000
    max_episode_length = 200
    best_reward = -np.inf
    for k in (range(max_num_episodes)):
        episode_cost = 0
        episode_steps = 0
        for i in range(max_episode_length):
            action = agent.choose_action(state, reparameterize=False)
            next_state, cost, terminated, truncated, _ = env.step(action)

            agent.remember((state, action, cost, next_state, terminated))

            state = next_state

            episode_cost += cost
            episode_steps += 1

            if terminated:
                break

        reward_arr.append(episode_cost)

        avg_reward = np.mean(reward_arr[-100:])

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save()
            print(f"Best model saved at episode {k} with average reward {avg_reward}")

        state, _ = env.reset()
        
        for j in range(episode_steps):
            if modelType == "lsac":
                agent.learn_lyapunov()
            agent.train()

        print(f"Episode {k} - Cost: {episode_cost}")
        if modelType == "lsac":
            print(f"Beta: {agent.beta.item()}")
    
    env.close()
    # agent.save()

if __name__ == "__main__":
    train_inverted_pendulum("lsac")
