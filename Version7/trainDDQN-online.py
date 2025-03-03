import os
import gym
import numpy as np
import argparse
from DDQN import DDQN
from wenv import CustomEnv
from utils import plot_learning_curve, create_directory, rns_reward

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='/home/nlsde/RLmodel/Version7/w2_baseline/online/')
parser.add_argument('--reward_path', type=str, default='/home/nlsde/RLmodel/Version7/w2_baseline/online/wB_avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='/home/nlsde/RLmodel/Version7/w2_baseline/online/wB_avg_epsilon.png')

args = parser.parse_args()

def ppo_train():
    env = CustomEnv()
    print(env.observation_space.shape[0], env.action_space.n)
    agent = DDQN(alpha=0.0003,state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, gamma=0.9)
    max_ep_len = 500                                 # max timesteps in one episode
    max_training_timesteps = int(1e3)   # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                 # update policy every n timesteps
    # update_timestep = 1000 
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []
    create_directory(args.ckpt_dir, sub_dirs=['model'])

    done = False
    observation = env.reset(seed=19)
    observation = observation[0]

    print(max_training_timesteps)

    while time_step <= max_training_timesteps:
        action = agent.choose_action(observation)
        observation_, reward, done, _, _, _ = env.step(action)
        agent.remember(observation, action, reward, observation_, done)

        time_step += 1
        
        # break; if the episode is over
        if done:
            print("train stop")
            break
            
        # update PPO agent
        if time_step % update_timestep == 0:
            agent.learn()
        observation = observation_
        

        i_episodes += 1
        episodes.append(i_episodes)
        total_rewards.append(reward * 500)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        print('EP:{:<5} reward:{:<20} avg_reward:{:<20} time_step:{:>10}'.
              format(i_episodes, reward * 500, avg_reward, time_step))
    
    # agent.save(args.ckpt_dir + '/model/ddqn_{}.pth'.format(i_episodes))
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    rns_reward(args.ckpt_dir+'ddqn_wB_reward_online.pkl', 'save', avg_rewards)
    return i_episodes, avg_rewards

if __name__ == "__main__":
    ppo_train()
    # validate()
            
        