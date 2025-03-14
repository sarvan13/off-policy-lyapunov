import gymnasium as gym
from env.quad import QuadRateEnv
import cv2
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def test_custom_inverted_pendulum(modelType, isTrajectory=True):
    # Create an instance of the custom environment
    if isTrajectory:
        env = gym.make("Quadrotor-v1")
    else:
        env = gym.make("Quadrotor-Still-v1")
    
    # Reset the environment to get the initial state
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    if modelType == "ppo":
        if isTrajectory:
            model = PPO.load("ppo_traj_quadrotor")
        else:
            model = PPO.load("ppo_quadrotor")
    
    elif modelType == "sac":
        if isTrajectory:
            model = SAC.load("sac_traj_quadrotor")
        else:
            model = SAC.load("sac_quadrotor")
    else:
        raise ValueError("Invalid model type")
    position_arr = []

    # Run the environment for 100 steps (or until termination)
    for step in range(1000):
        # env.render()  # Keep MuJoCo window open
        # print("Press any key to step... (ESC to exit)")
        
        # key = cv2.waitKey(0)  # Waits for key press while keeping GUI active
        # if key == 27:  # ESC key to exit
        #     break
        # Random action for testing
        # action = env.action_space.sample()

        # # Take a step in the environment
        # obs, reward, terminated, truncated, info = env.step(action)

        action, _ = model.predict(obs, deterministic=True)  # Get the best action
        obs, reward, terminated, truncated, info = env.step(action)
        pos = obs[0:3]
        position_arr.append(pos)

        # Accumulate the reward
        total_reward += reward

        # Print the results
        print(f"Step {step+1}:")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print("-" * 40)

        # Check if the episode is done
        if terminated or truncated:
            print(f"Episode finished after {step+1} steps.")
            break

    print(f"Total Reward: {total_reward}")
    if isTrajectory:
        ref_trajectory = env.unwrapped.reference_position
        print(ref_trajectory.shape)
    else:
        ref_trajectory = np.zeros((1000, 3))
    env.close()

    # Extract x, y, z coordinates from position_arr
    position_arr = np.array(position_arr)
    x = position_arr[:, 0]
    y = position_arr[:, 1]
    z = position_arr[:, 2]

    # Extract x, y, z coordinates from ref_trajectory
    ref_trajectory = np.array(ref_trajectory)
    ref_x = ref_trajectory[:, 0]
    ref_y = ref_trajectory[:, 1]
    ref_z = ref_trajectory[:, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o', label='Trajectory')

    # Plot the reference trajectory
    ax.plot(ref_x, ref_y, ref_z, color='blue', linestyle='--', label='Reference Trajectory')

    # Plot the initial point in green
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Initial Point')

    # Plot the final point in red
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='Final Point')

    # Set the range for all axes
    ax.set_xlim([0, 6])
    ax.set_ylim([-1, 5])
    ax.set_zlim([0, 5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(modelType + ' Quadrotor Trajectory')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    test_custom_inverted_pendulum("ppo", False)
