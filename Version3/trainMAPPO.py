import numpy as np
import argparse
from MAPPO import MAPPO
from wenvMAPPO import CustomEnv
from utils import plot_learning_curve, create_directory
import random

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--ckpt_dir', type=str, default='/home/nlsde/RLmodel/Version3/exp1_baseline/')
parser.add_argument('--reward_path', type=str, default='/home/nlsde/RLmodel/Version3/exp1_baseline/EP10000avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='/home/nlsde/RLmodel/Version3/exp1_baseline/EP10000epsilon.png')

args = parser.parse_args()

def mappo_train():
    env = CustomEnv()
    print(f"State dim: {env.observation_space.shape[0]}, Action dim: {env.action_space.n}")
    agent = MAPPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, 
                  lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2)
    max_ep_len = 500  # Max timesteps in one episode
    max_training_timesteps = int(8e4)  # Max total training timesteps
    update_timestep = max_ep_len * 4  # Update policy every n timesteps
    time_step = 0
    i_episodes = 0
    total_rewards, avg_rewards, episodes = [], [], []

    # Create directories for saving models and results
    create_directory(args.ckpt_dir, sub_dirs=['MAPPO'])

    while time_step <= max_training_timesteps:
        total_reward = 0
        done = False
        # Reset environment and get initial observations
        observation, _ = env.reset()  

        # Initialize variables to store the last action and state
        last_action = None
        last_observation = None

        # Interaction loop for each episode
        for t in range(1, max_ep_len+1):
            # Select actions for agents
            actions = [agent.choose_action(observation[i],i) for i in range(4)]
            action = random.choice(actions)
            # Store the last action and observation
            last_action = action
            last_observation = observation
            next_observations, rewards, done_flags, infos = zip(*[env.step(action,i) for i in range(4)])

            # Remember each agent's experiences
            for i in range(4):
                agent.remember(rewards[i], done_flags[i], i)

            # Update timestep and total reward
            time_step += 1
            total_reward += np.mean(rewards)  # Average the reward of all agents

            # Update policy if necessary
            if time_step % update_timestep == 0:
                agent.update()

            if any(done_flags):  # If any agent's episode is done
                break

            # Update observations for the next timestep
            observation = next_observations

        i_episodes += 1
        episodes.append(i_episodes)
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        print(f'EP:{i_episodes} Reward:{total_reward} Avg Reward:{avg_reward} Time Step:{time_step}')

         # Print last action and last observation at the end of each episode
        if any(done_flags):
            print(f"Episode {i_episodes} finished!")
            print(f"Last observation: {last_observation}")
            print(f"Last action: {last_action}")

    # Save model checkpoint
    agent.save(f'{args.ckpt_dir}/MAPPO/ppo_{i_episodes}.pth')

    # Plot and save the learning curve
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    return i_episodes, avg_rewards


if __name__ == "__main__":
    mappo_train()