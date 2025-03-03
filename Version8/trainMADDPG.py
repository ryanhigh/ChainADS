# -*- coding: utf-8 -*-
import numpy as np 
import argparse
from MADDPG import MADDPG
from wenv import CustomEnv 
from utils import plot_learning_curve, create_directory, rns_reward
from buffer import ReplayBuffer
import time

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='/home/nlsde/RLmodel/Version8/exp5_baseline/')
parser.add_argument('--reward_path', type=str, default='/home/nlsde/RLmodel/Version8/exp5_baseline/EP10000avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='/home/nlsde/RLmodel/Version8/exp5_baseline/EP10000epsilon.png')

args = parser.parse_args()

def maddpg_train():
    env = CustomEnv()
    num_agents=4
    agents = MADDPG(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, lr_actor=0.03, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    max_ep_len = 500                                  # max timesteps in one episode
    max_training_timesteps = int(5e5)   # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                  # update policy every n timesteps
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []
    create_directory(args.ckpt_dir, sub_dirs=['maddpg'])

    while time_step <= max_training_timesteps:
        total_reward = 0
        done = False
        observation = env.reset() 
        observation = observation[0]

        # 初始化动作列表
        actions = [None] * num_agents
        actionindexs =  [None] * num_agents
        rewards = [None] * num_agents

        for t in range(1, max_ep_len + 1):
            # 选择每个代理的动作
            for i in range(num_agents):
                actions[i],actionindexs[i],_= agents.choose_action(observation[i], i)
                # 执行动作并获得奖励
                observation_, rewards[i], done, info = env.step(actionindexs[i],i)
                agents.remember(observation[i], actionindexs[i], observation_, rewards[i], done, i)

            # 从四个代理的动作中选择奖励最高的动作
            best_action_idx = np.argmax(rewards)
            bestaction = actions[best_action_idx]
            bestactionindex = actionindexs[best_action_idx] # 根据奖励选择最佳动作

            # 更新所有代理，无论选择哪个动作
            actions = [bestaction] * num_agents
            actionindexs = [bestactionindex] * num_agents
            for i in range(num_agents):
                observation_, rewards[i], done, info = env.step(bestactionindex,i)
                observation[i] = observation_

            time_step += 1
            total_reward += max(rewards)

            # 每隔一定的时间步更新代理
            if time_step % update_timestep == 0:
                agents.update()

            if done:
                break

        i_episodes += 1
        episodes.append(i_episodes)
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        print('EP:{} reward:{} avg_reward:{} time_step{}'.
              format(i_episodes, total_reward, avg_reward, time_step))
        
    # Save the models for each agent
    #for i, agent in enumerate(agents):
        #agent.save('{}/maddpg/maddpg_agent_{}.pth'.format(args.ckpt_dir, i))
    
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    return i_episodes, avg_rewards

if __name__ == "__main__":
    start_time = time.time()
    maddpg_train()
    end_time = time.time()
    training_time = end_time - start_time
    print("模型训练花费的时间: {:.2f} 秒".format(training_time))
