import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import mujoco
from algorithms.lsac.agent import LSACAgent
from clean_ly import Agent, make_env
import torch
# ==========================================
# 2. Setup and Reproducibility
# ==========================================
SEED = 2
np.random.seed(SEED)
equilibrium_state = torch.tensor([np.array([
    0.0,  # x
    0.0,  # sin(theta1)
    0.0,  # sin(theta2)
    1.0,  # cos(theta1)
    1.0,  # cos(theta2)
    0.0,  # x_dot
    0.0,  # theta1_dot
    0.0,  # theta2_dot
    0.0   # constraint force
])], dtype=torch.float)

agent_path = r"C:\Users\Sarvan\Desktop\School\UVIC\thesis\off-policy-lyapunov\data\struct\InvertedDoublePendulum-v5\beta_0.5\mu_0.001\seed_12"
# Initialize environment
env = gym.make("InvertedDoublePendulum-v5")
model_type = "LSAC"
if model_type == "LSAC":
    model = LSACAgent(state_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0],
                max_action=env.action_space.high, dt=env.unwrapped.dt,
                equilibrium_state=equilibrium_state,
                save_dir=agent_path)
    model.load()
elif model_type == "LY":
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.vector.SyncVectorEnv([lambda: env])




# Set seed for reproducible environment initialization
obs, info = env.reset(seed=SEED)

# ==========================================
# 3. Trajectory Collection
# ==========================================
tip_x = []
tip_y = []  # We will map MuJoCo's Z-axis to Y for a standard 2D plot

# Get the internal site ID for the tip of the second pole
# The official gymnasium XML for this env names the site "tip"
site_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, "tip")

done = False
steps = 0
max_steps = 1000 # Safety limit

while not done and steps < max_steps:
    # 1. Get action from the model
    action = model.choose_action(obs)
    
    # 2. Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 3. Extract the physical position of the tip
    # site_xpos contains global 3D coordinates [X, Y, Z]
    tip_pos = env.unwrapped.data.site_xpos[site_id]
    
    tip_x.append(tip_pos[0]) # X axis (Cart movement direction)
    tip_y.append(tip_pos[2]) # Z axis (Vertical direction in MuJoCo)
    
    steps += 1

env.close()

# ==========================================
# 4. Plotting
# ==========================================
plt.figure(figsize=(8, 8))

# Plot the trajectory
plt.plot(tip_x, tip_y, label='Tip Trajectory', color='blue', linewidth=1.5, alpha=0.8)

# Mark the start and end points
plt.scatter(tip_x[0], tip_y[0], color='green', s=100, label='Start', zorder=5)
plt.scatter(tip_x[-1], tip_y[-1], color='red', s=100, label='End', zorder=5)

# Formatting
plt.ylim(1, 1.2)
plt.title("Inverted Double Pendulum: Tip Position Over 1 Trajectory", fontsize=14)
plt.xlabel("X Position (m)", fontsize=12)
plt.ylabel("Y Position (m) [MuJoCo Z-axis]", fontsize=12)

# Using an equal aspect ratio is critical here so the geometry of the 
# pendulum's swing doesn't look stretched or skewed.
plt.gca().set_aspect('equal', adjustable='box')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()