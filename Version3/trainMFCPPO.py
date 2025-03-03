import numpy as np
import argparse
from MFCPPO import MFCPPO
from wenvMFCPPO import CustomEnv
from utils import plot_learning_curve, create_directory

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='/home/nlsde/RLmodel/Version3/exp2_baseline/')
parser.add_argument('--reward_path', type=str, default='/home/nlsde/RLmodel/Version3/exp2_baseline/EP10000avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='/home/nlsde/RLmodel/Version3/exp2_baseline/EP10000epsilon.png')

args = parser.parse_args()

def mfcppo_train():
    env = CustomEnv()
    print(env.observation_space.shape[0], env.action_space.n)
    agent = MFCPPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,lr_actor=0.0001, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    max_ep_len = 100                                  # max timesteps in one episode
    max_training_timesteps = int(8e4)   # break training loop if timeteps > max_training_timesteps
    update_timestep = max_ep_len * 4                  # update policy every n timesteps
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []
    create_directory(args.ckpt_dir, sub_dirs=['PPO'])

    while time_step <= max_training_timesteps:
        total_reward = 0
        done = False
        observation = env.reset()
        observation = observation[0]
        # Initialize variables to store the last action and state
        last_action = None
        last_observation = None

        for t in range(1, max_ep_len+1):
            action = agent.choose_action(observation)
            last_action = action
            last_observation = observation
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
        print('EP:{} reward:{} avg_reward:{} time_step{}'.
              format(i_episodes, total_reward, avg_reward, time_step))
        
        # Print last action and last observation at the end of each episode
        # if done:
        #     print(f"Episode {i_episodes} finished!")
        #     print(f"Last observation: {last_observation}")
        #     print(f"Last action: {last_action}")

    agent.save(args.ckpt_dir + '/PPO/ppo_{}.pth'.format(i_episodes))
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    return i_episodes, avg_rewards


if __name__ == "__main__":
    mfcppo_train()
            
        