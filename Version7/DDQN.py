import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np # type: ignore
from buffer import ReplayBuffer
 
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
 
class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()
 
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, action_dim)
 
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
 
    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
 
        q = self.q(x)
 
        return q
 
    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)
 
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))
 
 
class DDQN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim=64, fc2_dim=32, ckpt_dir="/home/nlsde/RLmodel/Version7/w2_baseline",
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-7,
                 max_size=1000000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.action_space = [i for i in range(action_dim)]
 
        self.q_eval = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=fc1_dim, fc2_dim=fc2_dim)
 
        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)
 
        self.update_network_parameters(tau=1.0)
 
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
 
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
 
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
 
    def choose_action(self, observation, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
        actions = self.q_eval.forward(state)
        action = T.argmax(actions).item()
 
        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)
 
        return action
 
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
 
    def learn(self):
        if not self.memory.ready():
            return
 
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = np.arange(self.batch_size)
 
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals, dtype=T.bool).to(device)
 
        with T.no_grad():
            q_ = self.q_eval.forward(next_states_tensor)
            next_actions = T.argmax(q_, dim=-1)
            q_ = self.q_target.forward(next_states_tensor)
            q_[terminals_tensor] = 0.0
            rewards_tensor = T.tensor(rewards_tensor, dtype=T.float32).to(device)
            q_ = q_.to(dtype=T.float32)
            self.gamma = T.tensor(self.gamma, dtype=T.float32)
            batch_idx = T.tensor(batch_idx, dtype=T.long).to(device)
            target = rewards_tensor + self.gamma * q_[batch_idx, next_actions]
        q = self.q_eval.forward(states_tensor)[batch_idx, actions]
 
        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()
 
        self.update_network_parameters()
        self.decrement_epsilon()

    # def save_models(self, episode):
    #     self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
    #     print('Saving Q_eval network successfully!')
    #     self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DDQN_Q_target_{}.pth'.format(episode))
    #     print('Saving Q_target network successfully!')

    # def load_models(self, episode):
    #     self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
    #     print('Loading Q_eval network successfully!')
    #     self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DDQN_Q_target_{}.pth'.format(episode))
    #     print('Loading Q_target network successfully!')
 
    def save(self, checkpoint_path):
        T.save(self.q_eval(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(T.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(T.load(checkpoint_path, map_location=lambda storage, loc: storage))