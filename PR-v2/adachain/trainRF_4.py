import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
import sys
import os

# 获取父文件夹路径
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder)
from wenv import CustomEnv

env = CustomEnv()

num_episodes = 500
batch_size = 20
gamma = 0.9

# 初始化随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)

# 初始化经验回放缓冲区
replay_buffer = []
total_rewards = []
avg_rewards = []


# 随机采样几组状态动作和奖励值
X_train = []
y_train = []
for _ in range(20):
    observation = env.reset(seed=42)
    observation = observation[0]
    action = env.action_space.sample()
    observation_, reward, done, _, _, _ = env.step(action)
    X_train.append(observation)
    y_train.append(reward)

rf_model.fit(X_train, y_train)

print("=========================== finish pre train! ===========================")

# 训练循环
for episode in range(num_episodes):
    state = env.reset(seed=42)
    state = state[0]
    done = False
    total_reward = 0
    time_steps = 0

    while not done:
        # 选择动作
        action_values = rf_model.predict([state])
        action = np.argmax(action_values)  # 选择最大Q值对应的动作

        # 执行动作
        next_state, reward, done, _, _, _ = env.step(action)

        # 存储经验
        replay_buffer.append((state, action, reward, next_state, done))
        total_reward += reward

        # 更新状态
        state = next_state
        time_steps += 1

        # 从缓冲区采样并训练
        if len(replay_buffer) >= batch_size and episode % 20 == 0:
            batch = random.sample(replay_buffer, batch_size)
            states = np.array([exp[0] for exp in batch])
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch])
            next_states = np.array([exp[3] for exp in batch])
            dones = np.array([exp[4] for exp in batch])

            # 计算目标Q值
            target_q_values = rewards + gamma * np.max(rf_model.predict(next_states)) * (1 - dones)

            # 更新随机森林模型
            rf_model.fit(states, target_q_values)
    
    total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards[-100:])
    avg_rewards.append(avg_reward)
        
    print('EP:{} reward:{} avg_reward:{} steps:{}'.
              format(episode, total_reward, avg_reward, time_steps))