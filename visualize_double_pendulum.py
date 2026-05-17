import gymnasium as gym
import torch
import numpy as np
import argparse
import pathlib

from algorithms.lsac.agent import LSACAgent

# Add command line arguments
parser = argparse.ArgumentParser(description='Visualize a trained LSAC agent')
parser.add_argument('--mu', type=float, default=0.1, help='Lyapunov regularization parameter')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
args = parser.parse_args()

# --- 1. Setup Environment and Agent ---
env_name = "InvertedDoublePendulum-v5"
modelType = "lsac"

# Create the environment with render_mode='human'
env = gym.make(env_name, render_mode='human')

# Define the equilibrium state for the inverted double pendulum
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

# --- 2. Construct Model Path and Load Agent ---
script_dir = pathlib.Path(__file__).parent.resolve()
data_path = (
    script_dir /
    "data" /
    env_name /
    modelType /
    f"mu_{args.mu}" /
    f"seed_{args.seed}"
)

# Check if the model directory exists
if not data_path.exists():
    print(f"Error: Model directory not found at {data_path}")
    print("Please make sure you have trained the model for the specified mu and seed.")
    exit()

# Initialize the LSAC agent
agent = LSACAgent(
    state_dims=env.observation_space.shape[0],
    action_dims=env.action_space.shape[0],
    max_action=env.action_space.high,
    dt=env.unwrapped.dt,
    equilibrium_state=equilibrium_state,
    save_dir=str(data_path.absolute()) # Pass the save directory
)

# Load the trained models
agent.load()
print(f"Loaded model from {data_path}")


# --- 3. Run and Render Two Episode ---
for i in range(2):
    state, info = env.reset(seed=args.seed)
    done = False
    episode_reward = 0

    # input("Press Enter to start the simulation...")

    while not done:
        # Render the environment
        # env.render() # render is called automatically in human mode

        # Choose action from the loaded agent
        action = agent.choose_action(state, reparameterize=False)

        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        state = next_state
        episode_reward += reward

print(f"Episode finished. Total reward: {episode_reward}")

# Close the environment
env.close()
