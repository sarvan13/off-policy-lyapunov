import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal

class PPOMemory:
    def __init__(self, length, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.advantages = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.next_vals = []
        self.dones = []
        self.size = length

        self.batch_size = batch_size

    def generate_batches(self):
        # if len(self.advantages) != len(self.rewards):
        #     AssertionError("Must call calculate_advantages ONCE before generating batches")
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.next_states),\
                np.array(self.dones),\
                np.array(self.advantages),\
                batches

    def store_memory(self, state, action, probs, vals, reward, next_state, next_val, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.next_vals.append(next_val)
        self.dones.append(done)
    
    def calculate_advantages(self, gamma, gae_lambda):
        last_gae_lam = 0
        self.advantages = np.zeros_like(self.rewards, dtype=np.float32)
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.next_vals[step]
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.vals[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.vals[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.vals = []

class ActorNet(nn.Module):
    def __init__(self, state_dims, action_dims, max_action, lr=3e-4, fc1_dims=256, fc2_dims=256, 
                 reparam_noise=1e-6, name='actor.pth', save_dir='data\\ppo'):
        super(ActorNet, self).__init__()
        self.lr = lr
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.reparam_noise = reparam_noise
        self.name = name
        self.save_path = os.path.join(save_dir, name)

        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.action_dims)
        # self.log_sigma = nn.Linear(self.fc2_dims, self.action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = ('cuda:0' if T.cuda.is_available() else 'cpu')
        self.max_action = T.tensor(max_action).to(self.device)
        self.initial_cov_var = T.full(size=(self.action_dims,), fill_value=0.5)
        self.cov_var = T.full(size=(self.action_dims,), fill_value=0.5)
        self.final_cov_var = 0.2
        self.cov_mat = T.diag(self.initial_cov_var).to(self.device)
        self.to(self.device)
    
    def forward(self, state):
        layer1 = T.relu(self.fc1(state))
        layer2 = T.relu(self.fc2(layer1))
        mean = self.mu(layer2)
        # log_std = self.log_sigma(layer2).clamp(-20,2)
        # std = T.exp(log_std)

        return mean
    
    def sample(self, state, reparameterize=True):
        mean = self.forward(state)
        normal = MultivariateNormal(mean, self.cov_mat)

        if reparameterize:
            sampled_action = normal.rsample()
        else:
            sampled_action = normal.sample()

        # tanh_action = T.tanh(sampled_action)
        # action = tanh_action * self.max_action
        # log_prob = normal.log_prob(sampled_action) - T.log(self.max_action*(1 - tanh_action.pow(2)) + self.reparam_noise)
        # log_prob = log_prob.sum(dim=1, keepdim=True)

        log_prob = normal.log_prob(sampled_action)

        return sampled_action, log_prob
    
    def get_log_prob(self, state, action):
        mean = self.forward(state)
        normal = MultivariateNormal(mean, self.cov_mat)

        # tanh_action = action / self.max_action
        # sampled_action = T.atanh(tanh_action)
        # log_prob = normal.log_prob(sampled_action) - T.log(self.max_action*(1 - tanh_action.pow(2)) + self.reparam_noise)
        # return log_prob.sum(dim=1, keepdim=True)

        return normal.log_prob(action)
    
    def decay_covariance(self, total_episodes):
        self.cov_var -= (self.initial_cov_var - self.final_cov_var) / total_episodes
        self.cov_mat = T.diag(self.cov_var).to(self.device)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_path)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_path))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            save_dir='data\\ppo', name='critic.pth'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(save_dir, name)
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class PPOAgent:
    def __init__(self, n_actions, input_dims, max_action, update_freq, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10, normalize_advantage=True, beta = 0.5, entropy_coeff = 0.001,
            target_kl = None, save_dir='data\\ppo'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.input_dims = input_dims
        self.beta = beta
        self.entropy_coeff = entropy_coeff
        self.target_kl = target_kl
        self.normalize_advantage = normalize_advantage

        self.actor = ActorNet(input_dims,n_actions, max_action, lr=alpha, save_dir=save_dir)
        self.critic = CriticNetwork(input_dims, alpha, save_dir=save_dir)
        self.lyapunov = CriticNetwork(input_dims, alpha, save_dir=save_dir,  name='lyapunov.pth')
        self.memory = PPOMemory(update_freq, batch_size)
       
    def remember(self, state, action, probs, vals, reward, next_state, done):
        if done:
            next_val = self.critic(T.tensor(next_state, dtype=T.float).to(self.actor.device))
        else:
            # We only use next val in calculating the advantage
            # It is only needed for the last step in the episode
            # we set other instances to 0 to avoid repetitive forward pass 
            next_val = T.tensor(0.0, dtype=T.float).to(self.actor.device)
        next_val = T.squeeze(next_val).item()
        self.memory.store_memory(state, action, probs, vals, reward, next_state, next_val, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.lyapunov.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.lyapunov.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)

        value = self.critic(state)
        action, log_prob = self.actor.sample(state, False)

        probs = T.squeeze(log_prob).item()
        action = action.cpu().detach().numpy()[0]
        value = T.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        actor_losses = []
        critic_losses = []
        advantage = np.zeros(len(self.memory.rewards), dtype=np.float32)
        for _ in range(self.n_epochs):
            approx_kl_divs = []
            mean_actor_loss = []
            mean_critic_loss = []
            continue_training = True
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, next_state_arr, dones_arr, adv_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                next_states = T.tensor(next_state_arr[batch], dtype=T.float).to(self.actor.device)
                advantages = T.tensor(adv_arr[batch]).to(self.actor.device)

                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = self.actor.get_log_prob(states, actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                adjusted_advantage = advantages
                if self.normalize_advantage:
                    # Normalize the adjusted advantages
                    adjusted_advantage = (adjusted_advantage - adjusted_advantage.mean()) / (adjusted_advantage.std() + 1e-8)
                    
                weighted_probs = adjusted_advantage * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*adjusted_advantage
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # entropy loss
                entropy = -T.mean(-new_probs)

                returns = advantages + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss + self.entropy_coeff*entropy

                # with T.no_grad():
                #     log_ratio = new_probs - old_probs
                #     approx_kl_div = T.mean((T.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                #     approx_kl_divs.append(approx_kl_div)

                # if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                #     continue_training = False
                #     # print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}")
                #     break
 
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                mean_actor_loss.append(actor_loss.item())
                mean_critic_loss.append(critic_loss.item())

            if not continue_training:
                break

            actor_losses.append(np.mean(mean_actor_loss))
            critic_losses.append(np.mean(mean_critic_loss))

        self.memory.clear_memory()        
        return np.mean(actor_losses), np.mean(critic_losses)     

    def calculate_advantages(self):
        self.memory.calculate_advantages(self.gamma, self.gae_lambda)  

