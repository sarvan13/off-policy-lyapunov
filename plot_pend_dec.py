import gymnasium as gym
import numpy as np
from algorithms.lsac.agent import LSACAgent
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import torch
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
lsac_dir = os.path.join(curr_dir, "data", "Pendulum-v1", "lsac", "seed_12")

env = gym.make('Pendulum-v1')
env.reset()
agent = LSACAgent(state_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0],
                max_action=env.action_space.high, dt=env.unwrapped.dt,
                equilibrium_state=torch.tensor([np.array([np.cos(0), np.sin(0), 0])], dtype=torch.float),
                save_dir=lsac_dir)

def plot3D(agent):
    with torch.no_grad():
        agent.load()
        theta_arr = np.linspace(-np.pi, np.pi, 100)
        theta_dot_arr = np.linspace(-2, 2, 100)

        theta_grid, theta_dot_grid = np.meshgrid(theta_arr, theta_dot_arr)
        lyapunov_vals = np.zeros_like(theta_grid)

        eq_state = agent.equilibrium_state
        eq_action, _ = agent.actor.forward(eq_state)
        equilibrium_lyapunov = agent.lyapunov.forward(eq_state, eq_action)
        print(equilibrium_lyapunov)

        for i, theta in enumerate(theta_arr):
            for j, theta_dot in enumerate(theta_dot_arr):
                state = np.array([np.cos(theta), np.sin(theta), theta_dot])
                action, _ = agent.actor.forward(torch.tensor([state], dtype=torch.float).to(agent.actor.device))
                lyapunov_val = agent.lyapunov(torch.tensor([state], dtype=torch.float).to(agent.actor.device), action)
                lyapunov_vals[j, i] += lyapunov_val.item()

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(theta_grid, theta_dot_grid, lyapunov_vals, cmap='viridis')

        ax.set_xlabel('Theta')
        ax.set_ylabel('Theta Dot')
        ax.set_zlabel('Lyapunov Value')
        ax.set_title('LSAC Lyapunov Function')

        plt.show()

def plot2D(agent):
    with torch.no_grad():
        agent.load()
        theta_arr = np.linspace(-np.pi, np.pi, 100)
        theta_dot_arr = np.linspace(-1, 1, 100)

        theta_grid, theta_dot_grid = np.meshgrid(theta_arr, theta_dot_arr)
        lyapunov_vals = np.zeros_like(theta_grid)
        decrease_vals = np.zeros_like(theta_grid)

        eq_state = agent.equilibrium_state
        eq_action, _ = agent.actor.forward(eq_state)
        equilibrium_lyapunov = agent.lyapunov.forward(eq_state, eq_action)
        print(equilibrium_lyapunov)

        for i, theta in enumerate(theta_arr):
            for j, theta_dot in enumerate(theta_dot_arr):
                state = np.array([np.cos(theta), np.sin(theta), theta_dot])
                env.unwrapped.state = [theta, theta_dot]
                action, _ = agent.actor.forward(torch.tensor([state], dtype=torch.float).to(agent.actor.device))
                obs, rew, terminated, truncated, info = env.step(action.cpu().numpy())
                obs = obs.flatten()
                next_action, _ = agent.actor.forward(torch.tensor([obs], dtype=torch.float).to(agent.actor.device))

                lyapunov_val = agent.lyapunov(torch.tensor([state], dtype=torch.float).to(agent.actor.device), action)
                next_ly_val = agent.lyapunov(torch.tensor([obs], dtype=torch.float).to(agent.actor.device), next_action)

                decrease_vals[j,i] = (next_ly_val - lyapunov_val).item()

                lyapunov_vals[j, i] += lyapunov_val.item()

                # if decrease_vals[j, i] > 0:
                #     print(f"Red Dot: Theta={theta}, Theta Dot={theta_dot}, Decrease Val={decrease_vals[j, i]}")


        # Define custom contour levels
        levels = np.concatenate([
            np.linspace(0, 2.5, 10),  # More contours between 0 and 2.5
            np.linspace(2.6, 40, 20)  # Fewer contours between 2.6 and 40
        ])

        # Create a 2D contour plot (lines only, no color filling)
        fig, ax = plt.subplots()
        contour_lines = ax.contour(theta_grid, theta_dot_grid, lyapunov_vals, levels=20, colors='black')

        # Add green points for negative decrease_vals and red points for positive decrease_vals
        negative_indices = decrease_vals < 0
        positive_indices = decrease_vals > 0.001

        ax.scatter(theta_grid[negative_indices], theta_dot_grid[negative_indices], color='green', label='Decrease < 0', s=3)
        ax.scatter(theta_grid[positive_indices], theta_dot_grid[positive_indices], color='red', label='Decrease > 0', s=3)


        ax.set_xlabel('Theta')
        ax.set_ylabel('Theta Dot')
        ax.set_title('LSAC Lyapunov Function (Contour Lines Only)')

        # # Add labels to the contour lines
        # ax.clabel(contour_lines, inline=True, fontsize=8)

        plt.show()


plot2D(agent)