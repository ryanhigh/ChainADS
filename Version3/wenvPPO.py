import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version3/src/small_action3.csv')
action_space = action_space_df.values.tolist()
state_space_df = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_12_processed.csv')
state_space_df = state_space_df.fillna(state_space_df.shift())
if state_space_df.isnull().values.any():
    state_space_df = state_space_df.fillna(state_space_df.shift())
state_space = state_space_df.values.tolist()

MAX_TIMESTEPS = 100

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(len(action_space))
        self.observation_space = spaces.Box(
            low=state_space_df.min().values,
            high=state_space_df.max().values,
            dtype=np.float32
        )
        self.TPSm = state_space_df['tps'].max()
        self.Lm = state_space_df['delay'].max()
        self.SEm = 1
        self.state = None
        self.g = 1 
        self.timeCounts = 0

    def reward(self, L_t, SE_t, TPS_t):
        r_t = 0.4 * np.exp(1 - L_t/self.Lm) + 0.2 * np.exp(SE_t/self.SEm) + 0.4 * np.exp(TPS_t/self.TPSm)
        return r_t

    def security(self, m, connection):
        cluttering_coefficient_lt = [0.00, 0.29, 0.24]
        connect_lt = ['ring', 'tree', 'random']
        for i in range(len(connect_lt)):
            if connect_lt[i] == connection:
                q = cluttering_coefficient_lt[i] + 2
                break

        se = self.g * pow(m, q)
        return se


    def step(self, action):
        period, gaslimit, transactionpool, connection, position = action_space[action]
        self.state = np.array(state_space[action])

        L_t = self.state[5]
        SE_t = self.security(2, connection)
        TPS_t = self.state[2]
        current_reward = self.reward(L_t, SE_t, TPS_t)
        self.timeCounts += 1
        if ((self.TPSm - TPS_t) / self.TPSm < 0.05) or self.timeCounts > MAX_TIMESTEPS: done=True
        else: done=False
        return self.state, current_reward, done, {}
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed) 
        sample_action =  self.action_space.sample()
        self.state = np.array(state_space[sample_action])
        self.timeCounts = 0
        return self.state, {}
