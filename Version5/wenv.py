import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from gymnasium.utils import seeding

# Read CSV files for action space and state data
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version3/src/small_action3.csv')
action_space = action_space_df.values.tolist()
state_df1 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_12_processed.csv')
state_df2 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_13_processed.csv')
state_df3 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_14_processed.csv')
state_df4 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_15_processed.csv')

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

def process_state_space(state_space_df1, state_space_df2, state_space_df3, state_space_df4, type):
    state_space = []  # Initialize the result list

    if type == 1:
        # Convert the DataFrame to a list of lists (from type 1)
        state_space = state_space_df1.values.tolist()
        return state_space
    
    elif type == 2:
        # Concatenate the DataFrames vertically (type 2)
        state_space0 = pd.concat([state_space_df1, state_space_df2, state_space_df3, state_space_df4], axis=1, ignore_index=True)
        state_space = state_space0.values.tolist()
        return state_space
    
    elif type == 3:
        # Compute the element-wise mean (type 3)
        state_space0 = (state_space_df1 + state_space_df2 + state_space_df3 + state_space_df4) / 4
        state_space = state_space0.values.tolist()
        return state_space
    
    elif type == 4:
        # Convert DataFrames to numpy arrays for KDE computation (type 4)
        state_1_np = state_space_df1.to_numpy()
        state_2_np = state_space_df2.to_numpy()
        state_3_np = state_space_df3.to_numpy()
        state_4_np = state_space_df4.to_numpy()  
        
        # Loop over each row (state) in the arrays
        for i in range(len(state_1_np)):
            # Extract the data from each state for this iteration
            data1 = state_1_np[i]
            data2 = state_2_np[i]
            data3 = state_3_np[i]
            data4 = state_4_np[i]  
            
            # Initialize a list to hold KDEs (or means) for each dimension
            kdelis = []
            
            # Loop over each of the 11 dimensions (assuming there are 11 dimensions)
            for dim in range(11):
                # Collect the data for the current dimension across all states
                dimension_data = np.array([data1[dim], data2[dim], data3[dim], data4[dim]])
                
                # Attempt to compute KDE for this dimension
                try:
                    kde = gaussian_kde(dimension_data)  # KDE works with 1D data
                    sampled_data = kde.resample(1).flatten()[0]
                except Exception as e:
                    # If KDE fails, take the mean of the dimension
                    sampled_data = np.mean(dimension_data)
                
                # Append the KDE or mean to the list
                kdelis.append(sampled_data)
            
            # Append the list of KDEs/means for this state to the state_space
            state_space.append(kdelis)
        return state_space

    elif type == 5:
        # Sample state space initialization
        state_space = []
        state_1_np = state_space_df1.to_numpy()
        state_2_np = state_space_df2.to_numpy()
        state_3_np = state_space_df3.to_numpy()
        state_4_np = state_space_df4.to_numpy() 
        
        # Create bin edges according to the space
        bins_info = {}
        
        # Loop through each feature (column) to calculate bin edges
        for col in range(11):  # Assuming 11 features
            # Get min and max of each feature across the 4 state spaces
            feature_min = min(np.min(state_1_np[:, col]), np.min(state_2_np[:, col]),
                              np.min(state_3_np[:, col]), np.min(state_4_np[:, col]))
            feature_max = max(np.max(state_1_np[:, col]), np.max(state_2_np[:, col]),
                              np.max(state_3_np[:, col]), np.max(state_4_np[:, col]))
            
            # Create bin edges for each feature
            bin_edges = np.linspace(feature_min, feature_max, 10 + 1)  # 10 bins, 11 edges
            
            # Store bin edges for each feature
            bins_info[f'feature_{col}'] = {
                'bin_edges': bin_edges
            }
        
        # Now process each sample (i-th data point)
        for i in range(len(state_1_np)):
            # Initialize observation matrix of size (10, 11)
            obs = np.zeros((10, 11))
            data1 = state_1_np[i]
            data2 = state_2_np[i]
            data3 = state_3_np[i]
            data4 = state_4_np[i] 
            
            for dim in range(11):  # Iterate over the 11 dimensions (features)
                # Get the bin indices for each state space for each feature (dimension)
                bin_index_1 = np.digitize(data1[dim], bins_info[f'feature_{dim}']['bin_edges']) - 1
                bin_index_2 = np.digitize(data2[dim], bins_info[f'feature_{dim}']['bin_edges']) - 1
                bin_index_3 = np.digitize(data3[dim], bins_info[f'feature_{dim}']['bin_edges']) - 1
                bin_index_4 = np.digitize(data4[dim], bins_info[f'feature_{dim}']['bin_edges']) - 1
                
                # Check if the bin index is within valid range (0 to 9)
                if 0 <= bin_index_1 < 10:
                    obs[bin_index_1, dim] += 1
                if 0 <= bin_index_2 < 10:
                    obs[bin_index_2, dim] += 1
                if 0 <= bin_index_3 < 10:
                    obs[bin_index_3, dim] += 1
                if 0 <= bin_index_4 < 10:
                    obs[bin_index_4, dim] += 1
            
            # Append the observation matrix for the i-th data point
            state_space.append(obs.flatten())
        
        return state_space

state_space = process_state_space(state_space_df1, state_space_df2, state_space_df3, state_space_df4, 2)

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
        self.state = np.array(state_space[action])
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
        # sample_action =  self.action_space.sample() 
        self.np_random, seed = seeding.np_random(seed)
        sample_action = self.np_random.integers(low=0, high=len(action_space)-1, size=1)[0]
        self.state = np.array(state_space[sample_action])
        self.timeCounts = 0
        return self.state, {}
