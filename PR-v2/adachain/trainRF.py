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

def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)

def rf_train():
    env = CustomEnv()
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    # print(state_dim, action_dim)

    agent = AdaChainLearningAgent(n_actions=action_dim)

    max_ep_len = 500                                  # max timesteps in one episode
    max_training_timesteps = int(1e3)                 # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                  # update policy every n timesteps
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []
    
    # 随机采样几组状态动作和奖励值
    observation = env.reset()
    observation = observation[0]
    for _ in range(100):
        action = np.random.choice(env.action_space.n)  # 随机选择一个动作  # 随机动作
        observation_, reward, done, info = env.step(action)
        agent.add_experience(observation, action, reward)
        observation = observation_

    # 训练模型
    agent.train_model()
    print("=========================== finish pre train! ===========================")

    while time_step <= max_training_timesteps: 
        total_reward = 0
        done = False
        observation = env.reset()
        observation = observation[0] 
        for t in range(1, max_ep_len+1):
            action = agent.select_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.add_experience(observation, action, reward)
            
            time_step += 1
            total_reward += reward

            # update agent
            if time_step % update_timestep == 0:
                agent.train_model()
            
            # break; if the episode is over
            if done:
                break
            observation = observation_
            
        i_episodes += 1
        episodes.append(i_episodes)
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)

        print('EP:{} reward:{} avg_reward:{} time_step{}'.
            format(i_episodes, total_reward, avg_reward, time_step))
    
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', "/home/nlsde/RLmodel/PR-v2/adachain/RF1_reward.png")
            
if __name__ == "__main__":
    rf_train()

