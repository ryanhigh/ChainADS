import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from buffer import ReplayBuffer
from wenv import converted_data

# T.cuda.empty_cache() 
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
#device = T.device("cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=1):
        super(ActorCritic, self).__init__()

        self.system_state_dim = state_dim
        self.node_state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.actor = nn.Sequential(
            nn.Linear(11, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(20, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def act(self, x):
        x = T.tensor(x, dtype=T.float32).to(device) 
        action_probs = self.actor(x)
        noise = T.normal(mean=0.0, std=0.3, size=action_probs.shape)
        noise = noise.to(device)
        action_probs = action_probs + noise
        action_probs = T.softmax(action_probs, dim=-1)
        actionindex = T.multinomial(action_probs, 1) 
        if actionindex.dim() == 0:
        # If scalar (single item), just get the corresponding action
            action = converted_data[actionindex.item()]
        else:
        # If batch, use a list comprehension to handle each index in the batch
            action = [converted_data[i.item()] for i in actionindex]
        return action, actionindex, action_probs
    
    def Critic(self, state, action):
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32).to(device) 
        if not isinstance(action, T.Tensor):
            action = T.tensor(action, dtype=T.float32).to(device) 
        x = T.cat([state, action], dim=-1)
        x = self.critic(x)
        return x
    
class MADDPG:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, num_agents=4, device='cuda'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.num_agents = num_agents
        self.device = device

        # Create multiple buffers (one for each agent)
        self.buffers = [ReplayBuffer() for _ in range(self.num_agents)]

        # Create policies (actor-critic) and optimizers for each agent
        self.policies = [ActorCritic(state_dim, action_dim).to(self.device) for _ in range(self.num_agents)]
        self.optimizers = [optim.Adam([
                            {'params': self.policies[i].actor.parameters(), 'lr': lr_actor},
                            {'params': self.policies[i].critic.parameters(), 'lr': lr_critic}
                        ]) for i in range(self.num_agents)]

        # Create target networks for each agent (for stable training)
        self.target_policies = [ActorCritic(state_dim, action_dim).to(self.device) for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.target_policies[i].load_state_dict(self.policies[i].state_dict())
        
        # Create the old policies for computing loss (used for both actors and critics)
        self.policy_olds = [ActorCritic(state_dim, action_dim).to(self.device) for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.policy_olds[i].load_state_dict(self.policies[i].state_dict())
        
        self.MseLoss = nn.MSELoss() 
    
    def choose_action(self, state, agent_id):
        with T.no_grad():
            state = T.FloatTensor(state).to(device)  # Ensure the state is a tensor
            action,actionindex,probs = self.policy_olds[agent_id].act(state)  
            return action,actionindex,probs
    
    def remember(self, state, action, nextstate, reward, is_done, agent_id):
        self.buffers[agent_id].states.append(state)
        self.buffers[agent_id].actions.append(action)
        self.buffers[agent_id].rewards.append(reward)
        self.buffers[agent_id].nextstates.append(nextstate)
        self.buffers[agent_id].is_terminals.append(is_done)
    
    def soft_update(self, local_model, target_model, tau=0.01):
        """
        Soft update the target network using the local model's parameters.
        
        :param local_model: The local (current) model (i.e., actor or critic)
        :param target_model: The target model (i.e., target actor or target critic)
        :param tau: The soft update coefficient, typically a small value (e.g., 0.01)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * local_param.data)

    def update(self):
        # For each agent, perform updates using its experiences
        for i in range(self.num_agents):
            # Sample a batch of experiences from the buffer for agent i
            batch = self.buffers[i].sample_batch()
            states, actions, next_states, rewards, dones = batch
            # Compute target Q value using target critic networks
            next_actions,next_actionindex,next_probs= self.target_policies[i].act(T.tensor(next_states).to(device))
            done_tensor = T.tensor([1 - int(done) for done in dones], dtype=T.float32).to(device)
            rewards_tensor = T.tensor(rewards, dtype=T.float32).to(device)

            # 确保 done_tensor 和 critic 输出具有相同的形状
            target_q = rewards_tensor + (self.gamma * self.target_policies[i].Critic(next_states, next_actions) * done_tensor)
            
            # Update critic
            self.optimizers[i].zero_grad()
            actionscritic = [converted_data[j] for j in actions]
            q_value = self.policies[i].Critic(states, actionscritic)
            target_q = target_q.mean(dim=1).view(-1, 1)
            critic_loss = self.MseLoss(q_value, target_q.detach())
            critic_loss.backward()
            self.optimizers[i].step()

            # Update actor
            self.optimizers[i].zero_grad()
            states = T.tensor(states).to(device)
            update, updateindex, updateprobs= self.policies[i].act(states)
            actionscritic2 = T.tensor(update).to(device)
            actor_loss = -self.policies[i].Critic(states, actionscritic2).mean()
            actor_loss.backward()
            self.optimizers[i].step()

            # Soft update target network
            self.soft_update(self.policies[i], self.target_policies[i])

    def save(self, checkpoint_path):
        for agent_id in range(self.num_agents):
            T.save(self.policy_olds[agent_id].state_dict(), '{}_agent_{}.pth'.format(checkpoint_path, agent_id))

    def load(self, checkpoint_path):
        for agent_id in range(self.num_agents):
            self.policy_olds[agent_id].load_state_dict(T.load('{}{}_agent_{}.pth'.format(checkpoint_path, '_', agent_id), map_location=lambda storage, loc: storage))
            self.policies[agent_id].load_state_dict(T.load('{}{}_agent_{}.pth'.format(checkpoint_path, '_', agent_id), map_location=lambda storage, loc: storage))