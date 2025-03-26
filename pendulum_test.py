import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from algorithms.sac.agent import SACAgent
from algorithms.lsac.agent import LSACAgent

def train_inverted_pendulum(modelType):
    # Create an instance of the custom environment
    env = gym.make("Pendulum-v1")
    
    # Reset the environment to get the initial state
    state, info = env.reset()
    done = False
    total_reward = 0
    
    if modelType == "sac":
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high)
    elif modelType == "lsac":
        agent = LSACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, dt=env.unwrapped.dt)    
    else:
        raise ValueError("Invalid model type")

    max_num_episodes = 1000
    max_episode_length = 200
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

        state, _ = env.reset()
        
        for j in range(episode_steps):
            if modelType == "lsac":
                agent.learn_lyapunov()
            agent.train()

        print(f"Episode {k} - Cost: {episode_cost} - Beta: {agent.beta}")

if __name__ == "__main__":
    train_inverted_pendulum("lsac")
