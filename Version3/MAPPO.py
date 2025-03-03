import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
from buffer import RolloutBuffer
from torch.distributions import Categorical 

T.cuda.empty_cache() 
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)  
                    )

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )
    
    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        # actor Net compute action probabilities of actions
        action_probs = self.actor(state)
        # print("Action probs: ", action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        # critic Net compute state value
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    
class MAPPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, num_agents=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.num_agents = num_agents
        
        # Create multiple buffers (one for each agent)
        self.buffers = [RolloutBuffer() for _ in range(self.num_agents)]

        # Create policies and optimizers for each agent
        self.policies = [ActorCritic(state_dim, action_dim).to(device) for _ in range(self.num_agents)]
        self.optimizers = [optim.Adam([
                            {'params': self.policies[i].actor.parameters(), 'lr': lr_actor},
                            {'params': self.policies[i].critic.parameters(), 'lr': lr_critic}
                        ]) for i in range(self.num_agents)]

        # Old policies for each agent
        self.policy_olds = [ActorCritic(state_dim, action_dim).to(device) for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.policy_olds[i].load_state_dict(self.policies[i].state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def choose_action(self, state, agent_id):
        with T.no_grad():
            state = T.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_olds[agent_id].act(state)
        
            self.buffers[agent_id].states.append(state)
            self.buffers[agent_id].actions.append(action)
            self.buffers[agent_id].logprobs.append(action_logprob)
            self.buffers[agent_id].state_values.append(state_val)

        return action.item()
    
    def remember(self, reward, is_done, agent_id):
        self.buffers[agent_id].rewards.append(reward)
        self.buffers[agent_id].is_terminals.append(is_done)

    def update(self):
        for agent_id in range(self.num_agents):
            # Monte Carlo estimate of returns for each agent
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.buffers[agent_id].rewards), reversed(self.buffers[agent_id].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards
            rewards = T.tensor(rewards, dtype=T.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # Convert list to tensor for the current agent
            old_states = T.squeeze(T.stack(self.buffers[agent_id].states, dim=0)).detach().to(device)
            old_actions = T.squeeze(T.stack(self.buffers[agent_id].actions, dim=0)).detach().to(device)
            old_logprobs = T.squeeze(T.stack(self.buffers[agent_id].logprobs, dim=0)).detach().to(device)
            old_state_values = T.squeeze(T.stack(self.buffers[agent_id].state_values, dim=0)).detach().to(device)

            # Calculate advantages
            advantages = rewards.detach() - old_state_values.detach()

            # Optimize policy for K epochs for the current agent
            for _ in range(self.K_epochs):
                # Evaluating old actions and values for the current agent
                logprobs, state_values, dist_entropy = self.policies[agent_id].evaluate(old_states, old_actions)

                # Match state_values tensor dimensions with rewards tensor
                state_values = T.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = T.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss  
                surr1 = ratios * advantages
                surr2 = T.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Final loss of clipped objective PPO
                loss = -T.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

                # Take gradient step for the current agent
                self.optimizers[agent_id].zero_grad()
                loss.mean().backward()
                self.optimizers[agent_id].step()

            # Copy new weights into old policy for the current agent
            self.policy_olds[agent_id].load_state_dict(self.policies[agent_id].state_dict())

            # Clear buffer for the current agent
            self.buffers[agent_id].clear()

    def save(self, checkpoint_path):
        for agent_id in range(self.num_agents):
            T.save(self.policy_olds[agent_id].state_dict(), f'{checkpoint_path}_agent_{agent_id}.pth')

    def load(self, checkpoint_path):
        for agent_id in range(self.num_agents):
            self.policy_olds[agent_id].load_state_dict(T.load(f'{checkpoint_path}_agent_{agent_id}.pth', map_location=lambda storage, loc: storage))
            self.policies[agent_id].load_state_dict(T.load(f'{checkpoint_path}_agent_{agent_id}.pth', map_location=lambda storage, loc: storage))