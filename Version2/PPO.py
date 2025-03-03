import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
from buffer import RolloutBuffer2
from torch.distributions import Categorical # 分类分布,在本设计中不涉及连续动作空间，所以使用离散空间即可

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)  # 输出一个概率分布
                    )

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
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
    

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 5e-4
        self.action_space = [i for i in range(action_dim)]
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer2()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = T.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def choose_action(self, state):
        with T.no_grad():
            state = T.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        
        self.buffer.states.append(state)
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
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = T.squeeze(state_values)


            # =================== test 重要性采样 ======================
            behavior_logprobs = old_logprobs.clone()
            importance_sampling_weights = T.exp(logprobs - behavior_logprobs)
            importance_sampling_weights = T.clamp(importance_sampling_weights, 0, 1)  # 防止数值不稳定

            # Apply importance sampling weights to advantages
            weighted_advantages = importance_sampling_weights * advantages
            # ===============================================================


            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = T.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss  
            surr1 = ratios * weighted_advantages                        # 加权优势
            surr2 = T.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * weighted_advantages # 加权优势

            # final loss of clipped objective PPO
            loss = -T.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # # =================== test 修正 PPO 收敛范围 ======================
            # # 行为克隆正则化项
            # bc_loss = -logprobs.mean()
            # loss += 0.001 * bc_loss

            # # 保守 Q 值估计
            # q_values = state_values
            # # print(q_values)
            # cql_penalty = T.logsumexp(q_values, dim=0) - q_values.mean()
            # loss += 0.5 * cql_penalty
            # # ===============================================================
            
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

