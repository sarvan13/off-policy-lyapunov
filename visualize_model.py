import gymnasium as gym
import numpy as np
from algorithms.lsac.agent import LSACAgent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random
import torch
import os

from env.quad import QuadStillEnv
from env.quad import QuadRateEnv
from clean_ly import Agent, make_env
# from clean_ppo import Agent, make_env
# from clean_lppo import Agent
# env_name = "Hopper-v4"
env_name = "Quadrotor-Still-v1"
# env_name = "Quadrotor-v1"

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

# env = gym.make(env_name, render_mode="human")
env = gym.make(env_name)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ClipAction(env)

env = gym.vector.SyncVectorEnv([lambda: env])

algorithm = "ly"
algorithm_dir = f"clean_{algorithm}"
seed = 4

curr_dir = os.path.dirname(os.path.abspath(__file__))
agent_path = r"C:\Users\Sarvan\Desktop\School\UVIC\thesis\off-policy-lyapunov\data\Quadrotor-Still-v1\clean_ppo\seed_2\clean_ppo.cleanrl_model.pth"
mean_path = r"C:\Users\Sarvan\Desktop\School\UVIC\thesis\off-policy-lyapunov\data\Quadrotor-Still-v1\clean_ppo\seed_2\mean.npy"
var_path = r"C:\Users\Sarvan\Desktop\School\UVIC\thesis\off-policy-lyapunov\data\Quadrotor-Still-v1\clean_ppo\seed_2\var.npy"


# curr_dir = os.getcwd()
# agent_path = os.path.join(curr_dir, "runs", "Hopper-v4__clean_ppo__1__1743787519", "clean_ppo.cleanrl_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent(env).to(device)


agent.load_state_dict(torch.load(agent_path, map_location=device))
agent.eval()

mean = np.load(mean_path)
var = np.load(var_path)

# w2 = agent.actor_mean[0].weight.data

# print(w2)

def normalize_state(state, mean, var):
    # Normalize the state using the mean and variance
    normalized_state = (state - mean) / np.sqrt(var + 1e-8)
    return normalized_state

done = False

state_arr = []

reward_arr = []

reference_trajectory = np.array(env.envs[0].unwrapped.trajectory)
env_vel = env.envs[0].unwrapped.vd

obs, info = env.reset(seed=1)
# print(obs)
# obs = normalize_state(obs, mean, var)
# print(obs)
# action, logprob, _, value = agent.get_action_and_value(torch.Tensor(obs).to(device))
# print(action)
# next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
# print(next_obs)
# print(obs[0][0:3])
init_pos = reference_trajectory[0][0:3]
init_obs = init_pos + obs[0][0:3]
init_vel = env_vel + obs[0][7:10]
full_state = np.concatenate((init_obs, obs[0][3:7], init_vel, obs[0][10:13]))
raw_state_arr = []

state_arr.append(init_obs)
raw_state_arr.append(full_state)

# print(state_arr)


for i in range(1):
    obs, info = env.reset(seed=1)
    total_reward = 0
    done = False
    while not done:
        n_obs = normalize_state(obs, mean, var)

        action, logprob, _, value = agent.get_action_and_value(torch.Tensor(n_obs).to(device))
        next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())

        unflattened_obs = np.array([info["obx"][0], info["oby"][0], info["obz"][0]])
        unflattened_vel = np.array([info["obvx"][0], info["obvy"][0], info["obvz"][0]])
        raw_state = np.concatenate((unflattened_obs, obs[0][3:7], unflattened_vel, obs[0][10:13]))
        if len(state_arr) == 100:
            print(unflattened_obs)
            # print(raw_state_arr)
        state_arr.append(unflattened_obs)
        raw_state_arr.append(raw_state)

        total_reward += reward
        dones = (terminated | truncated).flatten()
        done = dones[0]

        obs = next_obs

    # print(unflattened_obs)
    reward = info["episode"]["r"]
    reward_arr.append(reward[0])
    # print(reward)

print(np.mean(reward_arr))
env.close()


raw_state_arr = np.array(raw_state_arr)
# print(raw_state_arr.shape)
# raw_vx = raw_state_arr[:,7]
# plt.plot(raw_vx, label='vx')
# plt.plot(raw_state_arr[:,8], label='vy')
# plt.plot(raw_state_arr[:,9], label='vz')
# plt.legend()
# plt.show()

#np.save("ppo_trajectory.npy", raw_state_arr)

# Convert state_arr to a NumPy array for easier slicing
state_arr = np.array(state_arr)

# Extract the first three elements of the state for plotting
x = state_arr[:, 0]  # First element
y = state_arr[:, 1]  # Second element
z = state_arr[:, 2]  # Third element

xt = reference_trajectory[:, 0]  # First element
yt = reference_trajectory[:, 1]  # Second element
zt = reference_trajectory[:, 2]  # Third element

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='State Trajectory')
ax.plot(xt, yt, zt, label='Reference Trajectory', color='red')
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.set_zlabel('z', fontsize=18)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.tick_params(axis='z', labelsize=18)
# Set axis limits
ax.set_xlim(0, 3)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0, 2.5)

ax.view_init(elev=30, azim=45)

ax.legend(fontsize=18)
plt.tight_layout()

# ax.set_title(f"{algorithm} Quadrotor Trajectory Tracking")

# Show the plot
plt.show()


