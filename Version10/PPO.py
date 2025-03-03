import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
from buffer import RolloutBuffer
from torch.distributions import Categorical 
from feature import HybridModel

device = torch.device("cuda:0" if T.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, system_state_dim, node_state_dim, action_dim, hidden_dim=64, num_layers=1):
        super(ActorCritic, self).__init__()

        self.system_state_dim = system_state_dim
        self.node_state_dim = node_state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.Hybrid = HybridModel(6,5,4,4,16)

        self.actor = nn.Sequential(
            nn.Linear(70, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(70, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def act(self, system_state, node_states):

        # Process the node embeddings through recursive layers
        
        concatstate = self.Hybrid(system_state,node_states)

        # Compute action probabilities with the actor
        action_probs = self.actor(concatstate)

        # Sample action based on probabilities
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # Compute the state value with the critic
        state_val = self.critic(concatstate)

        return action.detach(), action_logprob.detach(), state_val.detach(), concatstate.detach()

    def evaluate(self, total_state, action):
        # Compute action probabilities with the actor
        action_probs = self.actor(total_state)

        # Compute action log probabilities
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # Compute state values with the critic
        state_values = self.critic(total_state)
        
        return action_logprobs, state_values, dist_entropy
    
class PPO:
    def __init__(self, system_state_dim, node_state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 5e-4
        self.action_space = [i for i in range(action_dim)]
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(system_state_dim, node_state_dim, action_dim).to(device)
        self.optimizer = T.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(system_state_dim, node_state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def choose_action(self, state):
        with T.no_grad():
            system_state = state[0] 
            system_state = T.FloatTensor(system_state).to(device)
            node_states = state[1:] 
            node_states = [T.FloatTensor(node_state).to(device) for node_state in node_states]
            action, action_logprob, state_val, total_state= self.policy_old.act(system_state, node_states)  

        self.buffer.states.append(total_state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        return action.item()
    
    def remember(self, reward, is_done):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_done)

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

       # convert list to tensor
        old_states = T.squeeze(T.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = T.squeeze(T.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = T.squeeze(T.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = T.squeeze(T.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for i in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = T.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = T.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = T.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -T.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        T.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(T.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(T.load(checkpoint_path, map_location=lambda storage, loc: storage))

