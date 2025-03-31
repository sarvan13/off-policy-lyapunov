from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from algorithms.sac.agent import SACAgent
from algorithms.lsac.agent import LSACAgent
from env.quad.quad_rotor_still import QuadStillEnv
import torch

def train_inverted_pendulum(modelType):
    # Create an instance of the custom environment
    # env = gym.make("Pendulum-v1")
    env = gym.make("Quadrotor-Still-v1")
    
    # Reset the environment to get the initial state
    state, info = env.reset()
    done = False
    total_reward = 0
    
    if modelType == "sac":
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high)
    elif modelType == "lsac":
        agent = LSACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, 
                          dt=env.unwrapped.dt, equilibrium_state=torch.zeros((1, env.observation_space.shape[0]), dtype=torch.float), 
                          save_dir="data/pendulum/lsac")    
    else:
        raise ValueError("Invalid model type")
    agent.save()

    global_step = 0
    episode_count = 0
    max_steps = 5_000_000
    update_freq = 1000
    num_updates = max_steps // update_freq
    ep_reward_arr = deque(maxlen=25)

    for k in (range(num_updates)):
        episode_cost = 0
        episode_steps = 0
        for i in range(update_freq):
            action = agent.choose_action(state, reparameterize=False)
            next_state, cost, terminated, truncated, _ = env.step(action)

            agent.remember((state, action, cost, next_state, terminated))

            state = next_state
            global_step += 1

            episode_cost += cost
            episode_steps += 1

            if terminated or truncated:
                ep_reward_arr.append(episode_cost)
                episode_count += 1
                episode_cost = 0
                state, _ = env.reset()

        for j in range(update_freq):
            if modelType == "lsac":
                agent.learn_lyapunov()
            agent.train()


        print(f"Step {global_step} - Avg Code: {np.mean(ep_reward_arr)}")
        if modelType == "lsac":
            print(f"Beta: {agent.beta.item()}")
    
    env.close()
    agent.save()

if __name__ == "__main__":
    train_inverted_pendulum("lsac")
