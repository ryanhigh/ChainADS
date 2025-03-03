import numpy as np # type: ignore
import argparse
from DDQN import DDQN # type: ignore
from wenv import CustomEnv # type: ignore
from utils import plot_learning_curve, create_directory, rns_reward
import time

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='/home/nlsde/RLmodel/Version7/w2_baseline/')
parser.add_argument('--reward_path', type=str, default='/home/nlsde/RLmodel/Version7/w2_baseline/avg_reward.png')

args = parser.parse_args()

def ddqn_train():
    env = CustomEnv()
    agent = DDQN(alpha=0.001,state_dim=11, action_dim=env.action_space.n, gamma=0.99)
    max_ep_len = 500                                  # max timesteps in one episode
    max_training_timesteps = int(6e6)   # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                  # update policy every n timesteps
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []
    create_directory(args.ckpt_dir, sub_dirs=['DDQN'])

    while time_step <= max_training_timesteps:
        total_reward = 0
        done = False
        observation = env.reset(seed=19)
        observation = observation[0]

        for t in range(1, max_ep_len+1):
            action = agent.choose_action(observation)
            observation_, reward, done, _, _, _ = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            
            time_step += 1
            total_reward += reward

            # update agent
            if time_step % update_timestep == 0:
                agent.learn()
            
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
        reward_std = np.std(avg_rewards[-100:])
        if i_episodes > 100:
            if reward_std < 1e-3:
                print('training ends')
                break
        
    #agent.save(args.ckpt_dir+ 'DDQN/DDQN_q_eval_{}.pth'.format(i_episodes))
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    rns_reward(args.ckpt_dir+'ddqn_wB_reward.pkl', 'save', avg_rewards)
    return i_episodes, avg_rewards

if __name__ == "__main__":
    start_time = time.time()
    ddqn_train()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"模型训练花费的时间: {training_time:.2f} 秒")
        