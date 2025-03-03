import numpy as np
import argparse
from PPO import PPO
from wenv import CustomEnv
import torch.nn as nn
import torch as T

w1 = '/home/nlsde/RLmodel/PR-v2/workload1/PPO-w3/ppo_6027.pth'
w2 = '/home/nlsde/RLmodel/PR-v2/workload2/PPO-w3/ppo_6698.pth'

env = CustomEnv()
agent1 = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
agent2 = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
agent1.load(w1)
agent2.load(w2)
# agent1.load(w3)

class Reshape():
    def __init__(self):
        """
        初始化Reshape类
        """
        self.weights = None  # 存储权重
        self.action_z = []  
        self.critic_z = []
        self.source_policy = None
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_agents(self, aglist):
        self.source_policy = aglist
    
    def get_z(self, network_id, layer_id, input_data):
        z_all = []
        for agent in self.source_policy:
            _ = agent.choose_action(input_data)
            actor_ouput = agent.policy_old.actor_outputs
            critic_output = agent.policy_old.critic_outputs
            if network_id == "actor":
                z = actor_ouput[layer_id - 1]
            elif network_id == "critic":
                z = critic_output[layer_id - 1]
            z_all.append(z)
        return np.array(z_all)

    
    def __call__(self, network_id, layer_id, z, state, p):
        """
        对输入z进行reshape操作
        Args:
            z:              输入张量
            network_id:     actor/critic
            layer_id:       层数编号
            state:          环境状态参数, 即输入数据 
            p:              温度因子
        Returns:
            重塑后的张量
        """
        if self.weights is None:
            raise ValueError("未设置权重, 请先使用set_weights设置权重")
        
        source_z = []
        for agent in self.source_policy:
            z_i = self.get_z(network_id, layer_id, agent, state)
            source_z.append(z_i)
        print(source_z)
        z_src = np.array(source_z)
        
        # print(self.weights)
        # 应用权重进行重塑
        # reshaped_z = (1 - p) * np.sum(z_src * self.weights) + z * p
        total = np.zeros(len(z))
        for i in range(len(source_z)):
            print("z_src: ", i+1, "value", z_src[i])
            total += np.array(z_src[i]) * self.weights[i]
        print(total)
        reshaped_z = total*(1-p)+p*z
        return reshaped_z
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.actor_fc1 = nn.Linear(state_dim, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, action_dim)
        self.reshape = Reshape()
        self.reshape.set_weights([0.5,0.5])
    
    def forward(self, state, isTrain=True):
        z1_actor = self.actor_fc1(state)
        z1_actor = T.tanh(z1_actor)
        z2_actor = self.actor_fc2(z1_actor)
        z2_actor = T.tanh(z2_actor)
        z3_actor = self.actor_fc3(z2_actor)
        z3_actor = T.softmax(z3_actor, dim=-1)
        return z3_actor

    
def main():
    env = CustomEnv()
    observation = env.reset(seed=19)
    obs = observation[0]
    agt_l = [agent1, agent2]

    reshape = Reshape()
    
    w = np.array([0.5, 0.5])
    reshape.set_weights(w)
    reshape.set_agents(agt_l)
    z = reshape.get_z(network_id="actor", layer_id=2, input_data=obs)
    print(f"network=actor, agent=workload1, layer=2, z={z}")
    # reshaped_z = reshape(network_id="critic", layer_id=3, z=z0, state=obs, p=0.8)
    # print(f"original z:{z0}\nreshaped_z:{reshaped_z}")
    # _ = agent1.choose_action(obs)
    # print(agent1)
    

if __name__ == "__main__":
    main()