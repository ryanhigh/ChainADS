import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch

# Read CSV files for action space and state data
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version3/src/small_action3.csv')
action_space = action_space_df.values.tolist()

state_df1 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_12_processed.csv')
state_df2 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_13_processed.csv')
state_df3 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_14_processed.csv')
state_df4 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_15_processed.csv')

#Deal with NaN
state_space_df1 = state_df1.fillna(state_df1.shift())
if state_space_df1.isnull().values.any():
    state_space_df = state_space_df1.fillna(state_space_df1.shift())
state_space_df2 = state_df2.copy()
state_space_df3 = state_df3.copy()
state_space_df4 = state_df4.copy()

for col in state_df2.columns:
    state_space_df2[col] = state_space_df2[col].fillna(state_space_df1[col])
    state_space_df3[col] = state_space_df3[col].fillna(state_space_df1[col])
    state_space_df4[col] = state_space_df4[col].fillna(state_space_df1[col])

state_space_df1 = state_space_df1.ffill()
state_space_df2 = state_space_df2.ffill()
state_space_df3 = state_space_df3.ffill()
state_space_df4 = state_space_df4.ffill()

state_1_list = state_space_df1.values.tolist()
state_2_list = state_space_df2.values.tolist()
state_3_list = state_space_df3.values.tolist()
state_4_list = state_space_df4.values.tolist()

MAX_TIMESTEPS = 100

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.num_agents = 4
        self.action_space = spaces.Discrete(len(action_space))  # Discrete action space
        self.observation_space = spaces.Box(
            low=state_df1.min().values,  # Define observation space based on state
            high=state_df1.max().values,
            dtype=np.float32
        )
        self.TPSm = state_df1['tps'].max()  # Max TPS value
        self.Lm = state_df1['delay'].max()  # Max delay value
        self.SEm = 1  # Security constant
        self.state = None
        self.g = 1  # Security constant
        self.timeCounts = 0

    def reward(self, L_t, SE_t, TPS_t):
        r_t = 0.4 * np.exp(1 - L_t/self.Lm) + 0.2 * np.exp(SE_t/self.SEm) + 0.4 * np.exp(TPS_t/self.TPSm)
        return r_t

    def security(self, m, connection):
        # Security function based on connection type
        cluttering_coefficient_lt = [0.00, 0.29, 0.24]
        connect_lt = ['ring', 'tree', 'random']
        for i in range(len(connect_lt)):
            if connect_lt[i] == connection:
                q = cluttering_coefficient_lt[i] + 2
                break
        se = self.g * pow(m, q)
        return se

    def step(self, action, agent_id):
        period, gaslimit, transactionpool, connection, position = action_space[action]
        # Set state based on action and agent_id
        if agent_id == 0:
            self.state = np.array(state_1_list[action]) 
        elif agent_id == 1:
            self.state = np.array(state_2_list[action]) 
        elif agent_id == 2:
            self.state = np.array(state_3_list[action]) 
        elif agent_id == 3:
            self.state = np.array(state_4_list[action]) 

        L_t = self.state[5]  # Delay value from state
        SE_t = self.security(2, connection)  # Compute security
        TPS_t = self.state[2]  # TPS value from state
        current_reward = self.reward(L_t, SE_t, TPS_t)
        self.timeCounts += 1

        # Check if the episode is done
        if ((self.TPSm - TPS_t) / self.TPSm < 0.05) or self.timeCounts > MAX_TIMESTEPS: done=True
        else: done=False
        return self.state, current_reward, done, {}

    def reset(self, seed=None):
        # Reset the environment to a random state
        sample_action = self.action_space.sample()
        # Reset state 
        self.state = [state_1_list[sample_action],state_2_list[sample_action],state_3_list[sample_action],state_4_list[sample_action]]
        self.timeCounts = 0
        return self.state, {}  # Return the reset state
