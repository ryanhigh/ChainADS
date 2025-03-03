import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Read CSV files for action space and state data
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version3/src/small_action3.csv')
action_space = action_space_df.values.tolist()
state_df1 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_12_processed.csv')
state_df2 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_13_processed.csv')
state_df3 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_14_processed.csv')
state_df4 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_15_processed.csv')

#Deal with NaN
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

# Convert DataFrame to numpy arrays for KDE computation
state_1_np = state_space_df1.to_numpy()
state_2_np = state_space_df2.to_numpy()
state_3_np = state_space_df3.to_numpy()
state_4_np = state_space_df4.to_numpy()

# Assuming state_1_np, state_2_np, state_3_np, state_4_np are numpy arrays of appropriate shapes
state_space = []

for i in range(len(state_1_np)):
    # Extract the data from each state for this iteration
    data1 = state_1_np[i]
    data2 = state_2_np[i]
    data3 = state_3_np[i]
    data4 = state_4_np[i]
    
    # Initialize a list to hold KDEs for each dimension
    kdelis = []
    
    # Loop over each of the 11 dimensions (1 to 10)
    for dim in range(11):
        # Collect the data for the current dimension across all states
        dimension_data = np.array([data1[dim], data2[dim], data3[dim], data4[dim]])
        
        # Reshape to 2D array (4 samples, 1 dimension)
        dimension_data_reshaped = dimension_data.reshape(1, -1)  # shape (1, 4)
        
        # Attempt to compute KDE for this dimension
        try:
            kde = gaussian_kde(dimension_data_reshaped) 
            data = kde.resample(1).flatten()
            sampled_data = data[0]
        except Exception as e:
            sampled_data = np.mean(dimension_data)
        
        # Append the KDE or mean to the list
        kdelis.append(sampled_data)
    
    # Append the list of KDEs/means for this state to the state_space
    state_space.append(kdelis)

state_space_df = pd.DataFrame(state_space)

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
        self.TPSm = state_df1['tps'].max()
        self.Lm = state_df1['delay'].max()
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
        sample_action =  self.action_space.sample()
        self.state = np.array(state_space[sample_action])
        self.timeCounts = 0
        return self.state, {}
