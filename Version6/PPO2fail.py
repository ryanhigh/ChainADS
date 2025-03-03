import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
from buffer import RolloutBuffer
from torch.distributions import Categorical 

device = torch.device("cuda:0" if T.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, system_state_dim, node_state_dim, action_dim, node_embedding_dim=64):
        super(ActorCritic, self).__init__()

        # System-level state input size (e.g., 8)
        self.system_state_dim = system_state_dim

        # Node-level state input size (e.g., 3 for each node-level state)
        self.node_state_dim = node_state_dim

        # Define the embedding for the node-level states
        self.node_embedding = nn.Sequential(
            nn.Linear(node_state_dim, node_embedding_dim),  # Embedding each node-level state
            nn.Tanh(),
            nn.Linear(node_embedding_dim, node_embedding_dim)
        )

        # Total input dimension for both system-level and node-level states after embedding
        total_state_dim = system_state_dim + 4 * node_embedding_dim  # 4 nodes, each with node_embedding_dim size

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(total_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # Action probability distribution
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(total_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Value function output
        )

    def forward(self):
        raise NotImplementedError

    def act(self, system_state, node_states):
        # Process node-level states through embedding
        node_embeddings = [self.node_embedding(node_state) for node_state in node_states]
        node_embeddings = torch.stack(node_embeddings, dim=0)  # Shape: (4, node_embedding_dim)

        # Flatten the node embeddings and concatenate with system-level state
        node_embeddings_flat = node_embeddings.view(-1)  # Flatten to a 1D tensor
        total_state = torch.cat((system_state, node_embeddings_flat), dim=-1)

        # Compute action probabilities with the actor
        action_probs = self.actor(total_state)

        # Sample action based on probabilities
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # Compute the state value with the critic
        state_val = self.critic(total_state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, old_system_states, old_node_states, old_actions):
        action_logprobs_list = []
        state_values_list = []
        dist_entropy_list = []
        oldsystem_states = old_system_states.tolist()
        oldnode_states = old_node_states.tolist()
        oldactions = old_actions.tolist()
        for i in range(len(old_node_states)):
            node_embeddings = [self.node_embedding(torch.tensor(node_state).to(device)) for node_state in oldnode_states[i]]
            node_embeddings = torch.stack(node_embeddings, dim=0).to(device)
            print(f"node_embeddings device: {node_embeddings.device}")

            # Flatten node embeddings and concatenate with system state
            node_embeddings_flat = node_embeddings.view(-1)
            total_state = torch.cat((torch.tensor(oldsystem_states[i]).to(device), node_embeddings_flat), dim=-1).to(device)

            # Ensure action probabilities are computed with float64
            action_probs = self.actor(total_state)
            dist = Categorical(action_probs)

            # Calculate action log probabilities
            action_logprobs = dist.log_prob(torch.tensor(oldactions[i]).to(device))

            # Compute entropy of the action distribution
            dist_entropy = dist.entropy()

            # Compute state value from the critic network
            state_values = self.critic(total_state)

            # Append results for this batch element
            action_logprobs_list.append(action_logprobs)
            state_values_list.append(state_values)
            dist_entropy_list.append(dist_entropy)

        # Convert lists of results into tensors for batch processing
        action_logprobs = torch.stack(action_logprobs_list)
        state_values = torch.stack(state_values_list)
        dist_entropy = torch.stack(dist_entropy_list)

        # Return the results
        return action_logprobs, state_values, dist_entropy
    
class PPO:
    def __init__(self, system_state_dim, node_state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, node_embedding_dim=64):
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 5e-4
        self.action_space = [i for i in range(action_dim)]
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(system_state_dim, node_state_dim, action_dim, node_embedding_dim).to(device)
        self.optimizer = T.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(system_state_dim, node_state_dim, action_dim, node_embedding_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def choose_action(self, state):
        with T.no_grad():
            system_state = state[0] 
            system_state = T.FloatTensor(system_state).to(device)
            node_states = state[1:] 
            node_states = [T.FloatTensor(node_state).to(device) for node_state in node_states]
            action, action_logprob, state_val= self.policy_old.act(system_state, node_states)  

        self.buffer.states.append(state)
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

        # Split each state in self.buffer.states into old_system_state and old_node_states
        old_system_states = []
        old_node_states = []

        # Iterate through each state and split
        for state in self.buffer.states:
            old_system_state=  T.tensor(state[0])
            old_node_state=  T.tensor(state[1:])
            old_system_states.append(old_system_state)
            old_node_states.append(old_node_state)

        old_system_states = T.squeeze(T.stack(old_system_states, dim=0)).detach().to(device)
        old_node_states = T.squeeze(T.stack(old_node_states, dim=0)).detach().to(device)
        old_actions = T.squeeze(T.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = T.squeeze(T.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = T.squeeze(T.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for i in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_system_states, old_node_states, old_actions)

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