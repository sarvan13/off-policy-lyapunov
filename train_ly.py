import gymnasium as gym
import numpy as np
from algorithms.ly.ly import LYAgent
from env.quad import QuadStillEnv
import time

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    N = 2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    agent = LYAgent(n_actions=env.action_space.shape[0], batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, dt=env.unwrapped.dt,
                    input_dims=env.observation_space.shape[0],
                    max_action=env.action_space.high, update_freq=N,
                    entropy_coeff=0.001)
    n_steps = 200_000
    init_time = time.time()
    curr_time = time.time() - init_time

    score_history = []
    actor_loss = []
    critic_loss = []
    lyapunov_loss = []
    std = []

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
                # actor_loss.append(a_loss)
                # critic_loss.append(c_loss)
                # lyapunov_loss.append(l_loss)
                learn_iters += 10
            if step % 10_000 == 0:
                verbose_flag = True
                curr_time = time.time() - init_time
            observation = observation_
            agent.actor.decay_covariance(n_steps)
        
        score_history.append(score)
        # std.append(agent.actor.cov_var[0].item())
        avg_score = np.mean(score_history[-100:])

        if verbose_flag:
            print('episode', ep_count, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                    'time_steps', step, 'learning_steps', learn_iters, 'time', curr_time)
            verbose_flag = False
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
    
    
    
    # plt.plot(np.arange(len(score_history)), score_history)
    # plt.plot(np.arange(len(score_history)), 1000*np.array(std))
    # plt.xlabel("Episodes")
    # plt.ylabel("Total Reward")
    # plt.show()

    # loss_steps = N*np.arange(len(actor_loss))
    # plt.plot(loss_steps, actor_loss)
    # plt.xlabel("Time Steps")
    # plt.ylabel("Policy Loss")
    # plt.show()

    # plt.plot(loss_steps, critic_loss)
    # plt.xlabel("Time Steps")
    # plt.ylabel("Value Loss")
    # plt.show()
    
    # plt.plot(loss_steps, lyapunov_loss)
    # plt.xlabel("Time Steps")
    # plt.ylabel("Lyapunov Loss")
    # plt.show()

    agent.save_models()
    np.save('ly2-reward-batch.npy', score_history)
