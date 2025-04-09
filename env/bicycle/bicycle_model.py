import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class KinematicBicycleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_deviation=5.0):
        super().__init__()
        
        # Vehicle parameters
        self.lw = 2.7  # Wheelbase length
        self.lfo = 0.9
        self.lro = 0.9
        self.l = self.lw + self.lfo + self.lro  # Total vehicle length
        self.dt = 0.1  # Time step
        self.max_steer = 0.52 # Max steering angle
        self.max_accel = 4.5  # Max acceleration (m/s^2)
        self.max_speed = 40.0  # Max speed (m/s)

        self.max_deviation = max_deviation
        self.current_step = 0
        
        # State: [x, y, theta, velocity]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, 0.0]),
            high=np.array([np.inf, np.inf, np.pi, self.max_speed]),
            dtype=np.float32
        )
        
        # Action: [acceleration, steering angle]
        self.action_space = spaces.Box(
            low=np.array([-self.max_accel, -self.max_steer]),
            high=np.array([self.max_accel, self.max_steer]),
            dtype=np.float32
        )
        
        # Target trajectory (for now, a simple straight line at y=0)
        self.target_traj = lambda x: 0  # y = f(x)
        
        self.state = None
        self.render_mode = render_mode
        
        # Visualization setup
        self.fig, self.ax = None, None
        self.trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Start at origin
        self.trajectory = [self.state[:2]]
        return self.state, {}

    def step(self, action):
        accel, delta = np.clip(action, self.action_space.low, self.action_space.high)
        x, y, theta, v = self.state
        
        # Update state using kinematic bicycle model
        dx, dy, dtheta, dv = self.compute_derivative(self.state, action)
        x += dx * self.dt
        y += dy * self.dt
        theta += dtheta * self.dt
        v += dv * self.dt
        
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # Normalize angle

        self.state = np.array([x, y, theta, v], dtype=np.float32)
        self.trajectory.append([x, y, theta])
        
        # Compute reward (tracking error)
        y_target = self.target_traj(x)
        tracking_error = abs(y - y_target)
        rew_alive = 1e-1
        reward = -tracking_error*1e-1 + rew_alive  # Penalize deviation from trajectory
        
        # Termination condition
        terminated = tracking_error > self.max_deviation

        if terminated:
            reward = -1000.0  # Large penalty for exceeding max deviation
        
        return self.state, reward, terminated, False, {}
    
    def compute_derivative(self, state, action):
        x, y, theta, v = state
        accel, delta = action
        
        beta = np.atan((self.l/2 - self.lro)/self.lw * np.tan(delta))
        # Compute derivatives using kinematic bicycle model
        dx = v * np.cos(theta + beta)
        dy = v * np.sin(theta + beta)
        dtheta = (v / self.lw) * np.sin(delta)
        dv = accel
        
        return np.array([dx, dy, dtheta, dv], dtype=np.float32)


    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-10, 50)
            self.ax.set_ylim(-10, 10)
            self.ax.set_xlabel("X position")
            self.ax.set_ylabel("Y position")
            self.ax.set_title("Kinematic Bicycle Model Trajectory")
            
        self.ax.clear()
        
        # Plot trajectory
        self.ax.plot(*zip(*[(t[0], t[1]) for t in self.trajectory]), marker="o", markersize=3, linestyle="-", label="Vehicle Path")
        x_vals = np.linspace(-10, 50, 100)
        y_vals = [self.target_traj(x) for x in x_vals]
        self.ax.plot(x_vals, y_vals, "r--", label="Target Trajectory")
        
        # Draw vehicle representation
        if self.trajectory:
            x, y, theta = self.trajectory[-1]
            car_length = self.l * 2.0
            car_width = self.l * 0.75
            
            rear_x = x - (car_length / 2) * np.cos(theta)
            rear_y = y - (car_length / 2) * np.sin(theta)
            front_x = x + (car_length / 2) * np.cos(theta)
            front_y = y + (car_length / 2) * np.sin(theta)
            
            car_rect = plt.Rectangle((rear_x - car_width / 2, rear_y - car_width / 2), car_length, car_width, angle=np.degrees(theta), color="blue", alpha=0.6)
            self.ax.add_patch(car_rect)
        
        self.ax.legend()
        plt.pause(0.01)

    def close(self):
        if self.fig:
            plt.close(self.fig)
