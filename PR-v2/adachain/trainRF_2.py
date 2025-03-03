import pandas as pd
import numpy as np
from randomforestsample_njobs import AdaChainLearningAgent
import matplotlib.pyplot as plt
import sys
import os

# 获取父文件夹路径
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder)
from wenv import CustomEnv
from utils import plot_learning_curve, rns_reward, create_directory

def rf_train():
    env = CustomEnv()
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    # print(state_dim, action_dim)

    agent = AdaChainLearningAgent(n_actions=action_dim)

    max_ep_len = 50                                   # max timesteps in one episode
    max_training_timesteps = int(4e3)                 # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                  # update policy every n timesteps
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []
    create_directory('/home/nlsde/RLmodel/PR-v2/adachain/online-d/', sub_dirs=['model'])
    
    # 随机采样几组状态动作和奖励值
    observation = env.reset()
    observation = observation[0]
    for _ in range(20):
        action = np.random.choice(env.action_space.n)  # 随机选择一个动作  # 随机动作
        observation_, reward, done, _, _, _ = env.step(action)
        agent.add_experience(observation, action, reward)
        observation = observation_

    # 训练模型
    agent.train_model()
    print("=========================== finish pre train! ===========================")

    done = False
    observation = env.reset()
    observation = observation[0] 
    while time_step <= max_training_timesteps: 
        action = agent.select_action(observation)
        observation_, reward, done, _, _, _ = env.step(action)
        agent.add_experience(observation, action, reward)
        
        time_step += 1

        # update agent
        if time_step % update_timestep == 0:
            agent.train_model()
        
        observation = observation_
        
            
        i_episodes += 1
        episodes.append(i_episodes)
        total_rewards.append(reward * 500)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)

        print('EP:{} reward:{} avg_reward:{} time_step{}'.
            format(i_episodes, 500 * reward, avg_reward, time_step))
    
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', "/home/nlsde/RLmodel/PR-v2/adachain/RF_reward.png")
    agent.save_model('/home/nlsde/RLmodel/PR-v2/adachain/online-d/model/RFmodel_{}.pth'.format(i_episodes))
    rns_reward('/home/nlsde/RLmodel/PR-v2/adachain/online-d/adachain_reward.pkl', 'save', avg_rewards)
            
if __name__ == "__main__":
    rf_train()

