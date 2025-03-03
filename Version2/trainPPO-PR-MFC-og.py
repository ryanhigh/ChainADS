# this is a basic implementation of PiReuse within PPO.
# compared to origin PPO, this file add a new function ppo.

# Version2: input has been redirected to MFC

import os
import gym
import numpy as np
import argparse
from PPO import PPO
from wenv import CustomEnv
from utils import plot_learning_curve, create_directory, rns_reward

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='Version2/exp2.5_PRMFC/')
parser.add_argument('--reward_path', type=str, default='Version2/exp2.5_PRMFC/w2_prmfc_avg_reward_v2.png')
parser.add_argument('--epsilon_path', type=str, default='Version2/exp2.5_PRMFC/w2_prmfc_epsilon.png')

args = parser.parse_args()
w1 = '/home/nlsde/RLmodel/Version2/exp0_baseline/PPO/ppo_14945.pth'
Max_training_timesteps = int(2e5)

# def epsilon_greedy(self, x, y):
#         e = np.random.uniform(0, 1)
#         if e < 1 - (self.global_step-1) * 0.0005:
#             return np.random.randint(self.action_shape[0])
#         else:
#             return np.argmax(self.policy[x, y, :])

def policy_selection(W, temperature):
    probs = np.exp(temperature * W) / np.sum(np.exp(temperature * W))
    selected_index = np.random.choice(2, p=probs)
    return selected_index, probs

def ppo_train():
    env = CustomEnv()
    agent = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    max_ep_len = 500                                  # max timesteps in one episode
    max_training_timesteps = int(5e5)   # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                  # update policy every n timesteps
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes, epsilon_history = [], [], [], []
    create_directory(args.ckpt_dir, sub_dirs=['PPO-seed19-v2'])
    
    policy_lib = ['omega', w1]               # [agent] + [w1]
    temperature = 0   # 温度因子
    delta_tau = 0.00005     # 改变值
    psi = 0.7
    W = np.zeros(len(policy_lib))
    U = np.zeros(len(policy_lib))


    while time_step <= max_training_timesteps:
        # select policy
        selected_ind, probs = policy_selection(W, temperature)
        print(selected_ind, probs, 'W=', W, 'tau=', temperature, 'U=', U)
        selected_policy = policy_lib[selected_ind]

        total_reward = 0
        done = False
        observation = env.reset(seed=19)
        observation = observation[0]

        for t in range(1, max_ep_len+1):
            tag = 0
            # choose action
            if selected_ind == 0:
                action = agent.choose_action(observation)
                tag = 1
            else:
                if np.random.uniform(0, 1) < psi:
                    # use past policy
                    agent_past = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
                    agent_past.load(selected_policy)
                    action = agent_past.choose_action(observation)
                else:
                    # epsilon greedy pol
                    action, flag = agent.epsilon_choose_action(observation)
                    if flag: tag=1
                    agent.decrement_epsilon()

            observation_, reward, done, info = env.step(action)
            if tag == 1:
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
        
        gain = avg_reward
        W[selected_ind] = (W[selected_ind] * U[selected_ind] + gain) / (U[selected_ind] + 1)
        U[selected_ind] += 1

        temperature += delta_tau
        if temperature > 1:
            temperature = 1
        epsilon_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} time_step:{} epsilon:{}'.
              format(i_episodes, total_reward, avg_reward, time_step, agent.epsilon))
    
    agent.save(args.ckpt_dir + '/PPO-seed19-v2/ppo_{}.pth'.format(i_episodes))
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    plot_learning_curve(episodes, epsilon_history, 'Epsilon', 'eps', args.epsilon_path)
    rns_reward(args.ckpt_dir+'w2_reward_prmfc_v2.pkl', 'save', avg_rewards)
    return i_episodes, avg_rewards


if __name__ == "__main__":
    ppo_train()
            
        