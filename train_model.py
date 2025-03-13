# Run Cart Pole env
import torch

import gymnasium as gym
from env.quad import QuadRateEnv
from algorithms.sac.agent import SACAgent
from tqdm import tqdm
import numpy as np

environment = gym.make("Quadrotor-v1")
print(environment.action_space.high)
agent = SACAgent(environment.observation_space.shape[0], environment.action_space.shape[0], environment.action_space.high, batch_size=512)

state, info = environment.reset(seed=42)
# max_num_episodes = 5_000
max_episode_length = 1000
max_steps = 10_000_000
cost_arr = []
step_arr = []
steps_per_episode = []
total_steps = 0
log_cnt = 0
longest_episode = 0

lyapunov_loss = []
q_loss = []
v_loss = []
actor_loss = []
num_episodes = 0

while total_steps < max_steps:
    episode_cost = 0
    episode_steps = 0
    for i in range(max_episode_length):
        action = agent.choose_action(state, reparameterize=False)
        next_state, cost, terminated, truncated, _ = environment.step(action)

        agent.remember((state, action, cost, next_state, terminated))

        state = next_state

        episode_cost += cost
        episode_steps += 1
        total_steps += 1
        log_cnt += 1

        if terminated or truncated:
            break

    state, _ = environment.reset()
    num_episodes += 1

    if episode_steps > longest_episode:
        longest_episode = episode_steps
    
    if num_episodes % 2 == 0:
        for j in range(int(0.75*episode_steps)):
            losses = agent.train()

    if log_cnt >= 10_000:
       log_cnt = 0
       print(f"Total steps: {total_steps}")
       print(f"Total episodes: {num_episodes}")
       print(f"Cost of last episode: {episode_cost}")
       print(f"Longest episode: {longest_episode}")
    steps_per_episode.append(episode_steps)
    cost_arr.append(episode_cost)
    step_arr.append(total_steps)

np.save("sac-quadrotor-cost2-arr.npy", np.array(cost_arr))
np.save("sac-quadrotor-step2-arr.npy", np.array(step_arr))
np.save("sac-quadrotor-lyapunov-loss2-arr.npy", np.array(lyapunov_loss))
np.save("sac-quadrotor-v-loss2-arr.npy", np.array(v_loss))
np.save("sac-quadrotor-q-loss2-arr.npy", np.array(q_loss))
np.save("sac-quadrotor-actor-loss2-arr.npy", np.array(actor_loss))
agent.save()