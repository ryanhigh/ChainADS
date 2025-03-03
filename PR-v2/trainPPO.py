import os
import gym
import numpy as np
import argparse
from PPO import PPO
from wenv import CustomEnv
import pickle
import torch
from utils import plot_learning_curve, create_directory, rns_reward

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='/home/nlsde/RLmodel/PR-v2/workload3-zrefractor/') 
parser.add_argument('--reward_path', type=str, default='/home/nlsde/RLmodel/PR-v2/workload3-zrefractor/w3_v2k500_avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='/home/nlsde/RLmodel/PR-v2/workload3-zrefractor/w3_epsilon.png')

args = parser.parse_args()
w_history = ['/home/nlsde/RLmodel/PR-v2/workload1/PPO-w3/ppo_6027.pth',
             '/home/nlsde/RLmodel/PR-v2/workload2/PPO-w3/ppo_6698.pth']

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
    agent = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    max_ep_len = 500                                  # max timesteps in one episode
    max_training_timesteps = int(2e5)   # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                  # update policy every n timesteps
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []
    create_directory(args.ckpt_dir, sub_dirs=['PPO-w3'])

    agt_list = []
    for w in w_history:
        agent_o = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
        agent_o.load(w)
        agt_list.append(agent_o)
    reshape = Reshape()
    reshape.set_weights([0.5, 0.5])
    reshape.set_agents(agt_list)
    p = 0
    

    while time_step <= max_training_timesteps:
        total_reward = 0
        done = False
        observation = env.reset(seed=19)
        observation = observation[0]
        agent.get_temperature(p)
        for t in range(1, max_ep_len+1):
            if p < 1:
                ac_output = reshape.get_z(observation)
                agent.get_original_z(ac_output)
                isTrain = True
            else:
                isTrain = False
            action = agent.choose_action(observation, isTrain)
            observation_, reward, done, info = env.step(action)
            agent.remember(reward, done)
            
            time_step += 1
            total_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                agent.update()
            
            # break; if the episode is over
            if done:
                break
            observation = observation_

        i_episodes += 1
        episodes.append(i_episodes)
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        p += 0.005
        if p >= 1:
            p = 1
        print('EP:{} reward:{} avg_reward:{} time_step{} temperature:{}'.
              format(i_episodes, total_reward, avg_reward, time_step, p))
    
    agent.save(args.ckpt_dir + '/PPO-w3/ppo_{}.pth'.format(i_episodes))
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    rns_reward(args.ckpt_dir+'w3_reward.pkl', 'save', avg_rewards)
    return i_episodes, avg_rewards


if __name__ == "__main__":
    ppo_train()
            
        