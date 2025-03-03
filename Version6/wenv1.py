import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from gymnasium.utils import seeding

# Read CSV files for action space and state data
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version6/src/small_action3.csv')
action_space = action_space_df.values.tolist()
state_df1 = pd.read_csv('/home/nlsde/RLmodel/Version6/src/w3_res_12_processed.csv')
state_df2 = pd.read_csv('/home/nlsde/RLmodel/Version6/src/w3_res_13_processed.csv')
state_df3 = pd.read_csv('/home/nlsde/RLmodel/Version6/src/w3_res_14_processed.csv')
state_df4 = pd.read_csv('/home/nlsde/RLmodel/Version6/src/w3_res_15_processed.csv')

#Deal with NaN
state_space_df = state_df1.fillna(state_df1.shift())
if state_space_df.isnull().values.any():
    state_space_df = state_space_df.fillna(state_space_df.shift())
state_space = state_space_df.values.tolist()
state_space_df1 = state_df1.fillna(state_df1.shift())
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

def process_state_space(state_space_df1, state_space_df2, state_space_df3, state_space_df4):
    state_space = []
    
    for i in range(len(state_space_df1)):
        # Use .iloc to access columns by their index
        state_0 = [state_space_df1.iloc[i, j] for j in [1, 2, 3, 4, 5, 7]]
        state_1 = [state_space_df1.iloc[i, k] for k in [0, 6, 8, 9, 10]]
        state_2 = [state_space_df2.iloc[i, k] for k in [0, 6, 8, 9, 10]]
        state_3 = [state_space_df3.iloc[i, k] for k in [0, 6, 8, 9, 10]]
        state_4 = [state_space_df4.iloc[i, k] for k in [0, 6, 8, 9, 10]]
        
        # Append the combined states to the main state_space list
        state_space.append([state_0, state_1, state_2, state_3, state_4])
    
    return state_space

state_space = process_state_space(state_space_df1, state_space_df2, state_space_df3, state_space_df4)

MAX_TIMESTEPS = 100

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(len(action_space))
        self.observation_space = spaces.Box(
            low=state_space_df1.min().values,
            high=state_space_df1.max().values,
            dtype=np.float32
        )
        self.TPSm = state_space_df1['tps'].max()
        self.Lm = state_space_df1['delay'].max()
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
        selected_action = action_space[action]
        period, gaslimit, transactionpool, connection, position = selected_action
        #change
        self.state = state_space[action]
        # realstate
        self.realstate = state_space_df1.iloc[action]

        L_t = self.realstate[5]
        SE_t = self.security(2, connection)
        TPS_t = self.realstate[2]

        current_reward = self.reward(L_t, SE_t, TPS_t)

        self.timeCounts += 1
        if ((self.TPSm - TPS_t) / self.TPSm < 0.05) or self.timeCounts > MAX_TIMESTEPS: done=True
        else: done=False
        
        return self.state, current_reward, done, {}
    
    def reset(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        sample_action = self.np_random.integers(low=0, high=len(action_space)-1, size=1)[0]
        #state change
        self.state = state_space[sample_action]
        self.timeCounts = 0
        return self.state, {}
