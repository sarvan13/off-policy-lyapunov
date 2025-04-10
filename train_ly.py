import gymnasium as gym
import numpy as np
from algorithms.ly.ly import LYAgent
from env.quad import QuadStillEnv
from env.cartpole.cost_pend import CustomInvertedPendulumEnv
import time
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LYAgent with command line arguments')
    parser.add_argument('--N', type=int, default=2048, help='Update frequency')
    parser.add_argument('--n_steps', type=int, default=200_000, help='Number of steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    # env = gym.make('Quadrotor-Still-v1')
    env = gym.make('Pendulum-v1')
    N = args.N
    batch_size = args.batch_size
    n_epochs = 10
    alpha = 0.0003
    equilibrium_state = torch.tensor([np.array([np.cos(0), np.sin(0), 0])], dtype=torch.float)
    agent = LYAgent(n_actions=env.action_space.shape[0], batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, dt=env.unwrapped.dt,
                    input_dims=env.observation_space.shape[0],
                    max_action=env.action_space.high, update_freq=N,
                    equilibrium_state=equilibrium_state ,entropy_coeff=0.001)
    n_steps = args.n_steps
    init_time = time.time()
    curr_time = time.time() - init_time

    score_history = []

    learn_iters = 0
    avg_score = 0
    best_score = -2000
    step = 0
    ep_count = 0
    verbose_flag = False

    while step < n_steps:
        observation, _ = env.reset()
        done = False
        score = 0
        ep_count += 1
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, observation_, done)
            if step % N == 0:
                agent.calculate_advantages()
                l_loss = agent.train_lyapunov()
                a_loss, c_loss = agent.learn()
                learn_iters += 10
            if step % 10_000 == 0:
                verbose_flag = True
                curr_time = time.time() - init_time
            observation = observation_
            # agent.actor.decay_covariance(n_steps)
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if verbose_flag:
            print('episode', ep_count, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                    'time_steps', step, 'learning_steps', learn_iters, 'time', curr_time)
            verbose_flag = False
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

    np.save('ly-rewards.npy', score_history)
