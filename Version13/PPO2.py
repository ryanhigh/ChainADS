import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
from buffer import RolloutBuffer
from wenv import CustomEnv
from feature import HybridModel
from torch.distributions import Categorical # 分类分布,在本设计中不涉及连续动作空间，所以使用离散空间即可

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
weight = [0.98593637, 0.01406363]# [9.99999998e-01, 1.60522805e-09]

class Actor(nn.Module):
    def __init__(self, action_dim, o_data):
        super(Actor, self).__init__()
        self.p = 0.5
        self.o_data = o_data
        self.actor_fc1 = nn.Linear(70, 128)
        self.actor_fc2 = nn.Linear(128, action_dim)
    
    def forward(self, state, isTrain=True):
        if isTrain:
            z1_actor = self.actor_fc1(state)
            
            # debug
            z1_actor_detached = z1_actor.detach().to(device)  # 假设 z1_actor 是一个 PyTorch 张量
            o_data1_0_detached = T.tensor(self.o_data1[0], dtype=T.float32).detach().to(device)  # 确保是张量
            o_data2_0_detached = T.tensor(self.o_data2[0], dtype=T.float32).detach().to(device)  # 确保是张量

            z1_new = self.p * z1_actor_detached + (1-self.p)*(weight[0] * o_data1_0_detached + weight[1] * o_data2_0_detached)
            z1_new = z1_new.requires_grad_()

            z1_actor = T.tanh(z1_new)
            z2_actor = self.actor_fc2(z1_actor)

            z2_actor_detached = z2_actor.detach().to(device)  # 假设 z2_actor 是一个 PyTorch 张量
            o_data1_1_detached = T.tensor(self.o_data1[1], dtype=T.float32).detach().to(device)  # 确保是张量
            o_data2_1_detached = T.tensor(self.o_data2[1], dtype=T.float32).detach().to(device)  # 确保是张量

            z2_new = self.p * z2_actor_detached + (1-self.p)*(weight[0] * o_data1_1_detached + weight[1] * o_data2_1_detached)
            z2_new = z2_new.requires_grad_()

            z2_actor = T.softmax(z2_new, dim=-1)
        else:
            z1_actor = self.actor_fc1(state)
            z1_actor = T.tanh(z1_actor)
            z2_actor = self.actor_fc2(z1_actor)
            z2_actor = T.softmax(z2_actor, dim=-1)
        return z2_actor
    
    def update_odata(self, o_data):
        self.o_data1 = o_data[0]
        self.o_data2 = o_data[1]
    
    def update_p(self, p):
        self.p = p


class Critic(nn.Module):
    def __init__(self, o_data):
        super(Critic, self).__init__()
        self.p = 0.5
        self.o_data = o_data
        self.critic_fc1 = nn.Linear(70, 128)
        self.critic_fc2 = nn.Linear(128, 1)
    
    def forward(self, state, isTrain=True):
        if isTrain:
            z1_critic = self.critic_fc1(state)

            z1_critic_detached = z1_critic.detach().to(device)  # 假设 z1_actor 是一个 PyTorch 张量
            o_data1_0_detached = T.tensor(self.o_data1[0], dtype=T.float32).detach().to(device)  # 确保是张量
            o_data2_0_detached = T.tensor(self.o_data2[0], dtype=T.float32).detach().to(device)  # 确保是张量

            z1_new = self.p * z1_critic_detached + (1-self.p)*(weight[0] * o_data1_0_detached+ weight[1] * o_data2_0_detached)
            z1_new = z1_new.requires_grad_()
            
            z1_critic = T.tanh(z1_new)
            z2_critic = self.critic_fc2(z1_critic)

            z2_critic_detached = z2_critic.detach().to(device)  # 假设 z1_actor 是一个 PyTorch 张量
            o_data1_1_detached = T.tensor(self.o_data1[1], dtype=T.float32).detach().to(device)  # 确保是张量
            o_data2_1_detached = T.tensor(self.o_data2[1], dtype=T.float32).detach().to(device)  # 确保是张量

            z2_new = self.p * z2_critic_detached + (1-self.p)*(weight[0] * o_data1_1_detached + weight[1] * o_data2_1_detached)
            z2_new = z2_new.requires_grad_()
            return z2_new
        else:
            z1_critic = self.critic_fc1(state)
            z1_critic = T.tanh(z1_critic)
            z2_critic = self.critic_fc2(z1_critic)
        return z2_critic
    
    def update_odata(self, o_data):
        self.o_data1 = o_data[0]
        self.o_data2 = o_data[1]
    
    def update_p(self, p):
        self.p = p


class ActorCritic(nn.Module):
    def __init__(self, system_state_dim, node_state_dim, action_dim, hidden_dim=64, num_layers=1):
        super(ActorCritic, self).__init__()

        self.system_state_dim = system_state_dim
        self.node_state_dim = node_state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.Hybrid = HybridModel(6,5,4,4,16)
        self.o_data = None
        
        self.actor = Actor(action_dim, self.o_data)
        self.critic = Critic(self.o_data)

        # Store hook handles to remove them later
        self.hooks = []
        self.actor_outputs = []
        self.critic_outputs = []

    def forward(self):
        raise NotImplementedError
    
    def og_data(self, ac_output):
        a_out = []
        c_out = []
        for item in ac_output:
            a_out.append(item[0])
            c_out.append(item[1])

        self.actor.update_odata(a_out)
        self.critic.update_odata(c_out)

    def update_p(self, p):
        self.actor.update_p(p)
        self.critic.update_p(p)
    
    def register_hooks(self):
        # Register forward hooks for each layer in actor and critic
        def hook_fn(module, input, output):
            # Check if the module is an instance of nn.Linear
            if isinstance(module, nn.Linear):
                # print(f"Layer: {module.__class__.__name__}, Output shape: {output.shape}, Output values: {output}")
                if module in self.actor.modules():
                    # Store the output of each Linear layer in actor
                    self.actor_outputs.append(output.detach().cpu().numpy())
                elif module in self.critic.modules():
                    # Store the output of each Linear layer in critic
                    self.critic_outputs.append(output.detach().cpu().numpy())
        
        # Register hooks for actor layers
        for idx, layer in self.actor.named_children():
            hook_handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook_handle)
        
        # Register hooks for critic layers
        for idx, layer in self.critic.named_children():
            hook_handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook_handle)

    def remove_hooks(self):
        # Remove all hooks after use
        for hook_handle in self.hooks:
            hook_handle.remove()
        self.hooks = []

    def act(self, system_state, node_states, isTrain):
        # Process the node embeddings through recursive layers
        concatstate = self.Hybrid(system_state,node_states)

        # actor Net compute action probabilities of actions
        action_probs = self.actor(concatstate, isTrain)
        # print("Action probs: ", action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        # critic Net compute state value
        state_val = self.critic(concatstate, isTrain)

        return action.detach(), action_logprob.detach(), state_val.detach(), concatstate.detach()
    
    def evaluate(self, state, action, isTrain):
        action_probs = self.actor(state, isTrain)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state, isTrain)
        
        return action_logprobs, state_values, dist_entropy
    
    def __str__(self):
        return f"Structure of Model:\nActor:\n{self.actor}\nCritic:\n{self.critic}"
    

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
    
    def get_original_z(self, ac_output):
        self.policy.og_data(ac_output)
        self.policy_old.og_data(ac_output)
        # print("============= successfully set og_data!!! =================")

    def get_temperature(self, p):
        self.policy.update_p(p)
        self.policy_old.update_p(p)
    
    def choose_action(self, state, isTrain=True):
        with T.no_grad():
            system_state = state[0]
            system_state = T.FloatTensor(system_state).to(device)
            node_states = state[1:] 
            node_states = [T.FloatTensor(node_state).to(device) for node_state in node_states]
            
            # Register hooks to track layer outputs
            self.policy_old.register_hooks()

            action, action_logprob, state_val, total_state = self.policy_old.act(system_state, node_states, isTrain)

            # Remove hooks after action is chosen
            self.policy_old.remove_hooks()
        
        self.buffer.states.append(total_state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()
    
    def epsilon_choose_action(self, state):
        """
        Here is a new function in PPO, for epsilon greedy of pi-reuse
        """
        e = np.random.uniform(0, 1)
        if e < self.epsilon:
            flag=0
            return np.random.choice(self.action_space), flag
        else:
            flag=1
            return self.choose_action(state), flag

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    
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

        if old_state_values.shape != rewards.shape:
            print(old_state_values.shape, rewards.shape)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, False)

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
    
    def __str__(self):
        return f"{self.policy}"


