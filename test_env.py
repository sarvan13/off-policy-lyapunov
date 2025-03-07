import gymnasium as gym
from env.quad import QuadRateEnv
import cv2

def test_custom_inverted_pendulum():
    # Create an instance of the custom environment
    env = gym.make("Quadrotor-v1", render_mode="human")
    
    # Reset the environment to get the initial state
    obs, info = env.reset()
    done = False
    total_reward = 0

    # Run the environment for 100 steps (or until termination)
    for step in range(100):
        env.render()  # Keep MuJoCo window open
        print("Press any key to step... (ESC to exit)")
        
        key = cv2.waitKey(0)  # Waits for key press while keeping GUI active
        if key == 27:  # ESC key to exit
            break
        # Random action for testing
        action = env.action_space.sample()

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)

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
    env.close()

if __name__ == "__main__":
    test_custom_inverted_pendulum()
