import numpy as np
import argparse
from PPO import PPO
from wenv4 import CustomEnv
import pandas as pd

from utils import plot_learning_curve, create_directory, rns_reward

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='/home/nlsde/RLmodel/Version6/baseline/PPO-wC/')
parser.add_argument('--reward_path', type=str, default='/home/nlsde/RLmodel/Version6/baseline/PPO-wC/wC_avg_reward_test.png')

args = parser.parse_args()
def ppo_test():
    env = CustomEnv()
    print(env.observation_space.shape[0], env.action_space.n)
    agent = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    
    total_test_episodes = 1000                         # test episodes num
    run_num_pretrained = 1205                         # set this to load a particular checkpoint num
    # run_num_pretrained = 613 ## onlyTPS
    max_ep_len = 500                                # max timesteps in one episode
    plot_timesteps = 1                             # plot interval

    action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version6/src/workload2/2action.csv')
    action_space = action_space_df.values.tolist()

    #directory = args.ckpt_dir + 'model/'
    #checkpoint_path = directory + "ppo_{}.pth".format(run_num_pretrained)
    #print("loading network from : " + checkpoint_path)
    #agent.load(checkpoint_path)
    #print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    create_directory(args.ckpt_dir+'test/', sub_dirs=['reward', 'tps', 'delay', 'security'])

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        total_tps = 0
        total_delay = 0
        total_security = 0
        done = False
        observation = env.initial()
        sample_action = observation[1]
        observation = observation[0]
        selected_action = action_space[sample_action]
        period, gaslimit, transactionpool, connection, position, number = selected_action

        L_t = observation[5]
        SE_t = env.security(number, connection)
        TPS_t = observation[2]
        r_org = env.reward(L_t, SE_t, TPS_t)
        r = [r_org]
        time = [0]
        total_tps += TPS_t
        total_delay += L_t
        total_security += SE_t
        tps = [total_tps]
        delay = [total_delay]
        se = [total_security]

    
        for t in range(1, max_ep_len+1):
            action = agent.choose_action(observation)
            observation_, reward, done, TPS_t, L_t, SE_t = env.step(action)
            ep_reward += reward
            total_tps += TPS_t
            total_delay += L_t
            total_security += SE_t

            if t % plot_timesteps == 0:
                r.append(ep_reward/t)
                tps.append(total_tps/(t+1))
                delay.append(total_delay/(t+1))
                se.append(total_security/(t+1))
                time.append(t)
            
            # break; if the episode is over
            if done:
                break
            observation = observation_
        
        plot_learning_curve(time, r, 'Reward', 'r', args.ckpt_dir+'test/reward/'+'reward_wC_{}.png'.format(ep))
        plot_learning_curve(time, tps, 'TPS', 'tps', args.ckpt_dir+'test/tps/'+'tps_wC_{}.png'.format(ep))
        plot_learning_curve(time, delay, 'Delay', 'delay', args.ckpt_dir+'test/delay/'+'delay_wC_{}.png'.format(ep))
        plot_learning_curve(time, se, 'Security', 'security', args.ckpt_dir+'test/security/'+'se_wC_{}.png'.format(ep))
        
        # clear buffer
        agent.buffer.clear()
        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == "__main__":
    ppo_test()