import os
import gym
import numpy as np
import argparse
from PPO2 import PPO
from wenv import CustomEnv
import torch
from utils import plot_learning_curve, create_directory, rns_reward

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='/home/nlsde/RLmodel/Version13/exp2_baseline/online-c-pr/')
parser.add_argument('--reward_path', type=str, default='/home/nlsde/RLmodel/Version13/exp2_baseline/online-c-pr/wC_avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='/home/nlsde/RLmodel/Version13/exp2_baseline/online-c-pr/wC_avg_epsilon.png')

args = parser.parse_args()
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
    
    def get_z(self, input_data):
        z_all = []
        for agent in self.source_policy:
            _ = agent.choose_action(input_data, False)
            actor_output = agent.policy_old.actor_outputs
            critic_output = agent.policy_old.critic_outputs
            
            # 处理 actor_output，每个元素检查是否是张量
            actor_output_np = [
                elem.detach().cpu().numpy() if isinstance(elem, torch.Tensor) else np.array(elem)
                for elem in actor_output
            ]
            
            # 处理 critic_output，每个元素检查是否是张量
            critic_output_np = [
                elem.detach().cpu().numpy() if isinstance(elem, torch.Tensor) else np.array(elem)
                for elem in critic_output
            ]

            # 将转换后的 actor 和 critic 输出添加到 z_all
            z_all.append([actor_output_np, critic_output_np])
        # return np.array(z_all)
        return z_all


def ppo_train():
    env = CustomEnv()
    print(env.observation_space.shape[0], env.action_space.n)
    agent = PPO(system_state_dim=6, node_state_dim=5, action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    max_ep_len = 500                                 # max timesteps in one episode
    max_training_timesteps = int(8e4)   # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                 # update policy every n timesteps
    update_p_timesteps = 1000 
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []
    create_directory(args.ckpt_dir, sub_dirs=['model'])

    w_history = ['/home/nlsde/RLmodel/Version13/exp2_baseline/online-b-pr/model/ppo_80001.pth',
             '/home/nlsde/RLmodel/Version13/exp2_baseline/online-d-pr/model/ppo_80001.pth']
    agt_list = []
    for w in w_history:
        agent_o = PPO(system_state_dim=6, node_state_dim=5, action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
        agent_o.load(w)
        agt_list.append(agent_o)
    reshape = Reshape()
    reshape.set_weights([0.5, 0.5])
    reshape.set_agents(agt_list)
    p = 0

    done = False
    observation = env.reset(seed=42)
    observation = observation[0]
    # isTrain = False

    while time_step <= max_training_timesteps:
        agent.get_temperature(p)
        # if p < 1:
        #     ac_output = reshape.get_z(observation)
        #     agent.get_original_z(ac_output)
        #     isTrain = True
        # else:
        #     isTrain = False
        action = agent.choose_action(observation, isTrain=False)
        observation_, reward, done, _ = env.step(action)
        agent.remember(reward, done)
        
        time_step += 1
        
        # update PPO agent
        if time_step % update_timestep == 0:
            agent.update()
        
        observation = observation_
        

        i_episodes += 1
        episodes.append(i_episodes)
        total_rewards.append(reward * 500)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        p += 0.005
        if p >= 1:
            p = 1
        print('EP:{:<5} reward:{:<20} avg_reward:{:<20} time_step:{:<10} temperature:{:>10}'.
              format(i_episodes, reward * 500, avg_reward, time_step, p))
    
    agent.save(args.ckpt_dir + '/model/ppo_{}.pth'.format(i_episodes))
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    rns_reward(args.ckpt_dir+'ppo_pr_fusion_wC_reward.pkl', 'save', avg_rewards)
    return i_episodes, avg_rewards

if __name__ == "__main__":
    ppo_train()