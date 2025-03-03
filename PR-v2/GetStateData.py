"""
FOR TRAIN WEIGHT NETWORK
"weight net"
(input)  r1, r2
(output) w1, w2
while [r1, r2] gets from source policy Reward(a = Pi(s))
so this file to generate a list of state information
"""

import os
import gym
import numpy as np
import argparse
from PPO import PPO
from wenv import CustomEnv
import pickle

def rns_reward(fname, mode, data):
    if mode=='save':
        # 保存列表到文件
        # pkl文件
        with open(fname, 'wb') as file:
            pickle.dump(data, file)
    elif mode=='read':
        # 从文件中读取列表
        with open(fname, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data

def get_reward(env, agent, obs):
    action = agent.choose_action(obs)
    obs_, rwd, done, info = env.step(action)
    return rwd

def Go():
    env = CustomEnv()
    obeservation = env.reset(seed=19)
    obs = obeservation[0]
    obs_list = [obs]
    rwd_agent1_list = []
    rwd_agent2_list = []
    agent = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    agent2 = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    w1 = '/home/nlsde/RLmodel/Version2/exp0_baseline/PPO/ppo_14945.pth'
    w2 = '/home/nlsde/RLmodel/Version2/exp1_PR/PPO-lowT/ppo_15656.pth'
    agent.load(w1)
    agent2.load(w2)
    epochs = 1000
    for i in range(epochs):
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        rwd_agent1_list.append(reward)
        rwd_agent2_list.append(get_reward(env, agent2, obs))
        obs_list.append(obs_)
        obs = obs_
        
    _ = obs_list.pop()
    rns_reward('/home/nlsde/RLmodel/PR-v2/stateData.pkl', 'save', obs_list)
    rns_reward('/home/nlsde/RLmodel/PR-v2/stateReward1.pkl', 'save', rwd_agent1_list)
    rns_reward('/home/nlsde/RLmodel/PR-v2/stateReward2.pkl', 'save', rwd_agent2_list)

if __name__ == "__main__":
    Go()



