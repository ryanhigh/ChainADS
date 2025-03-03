import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore
import pandas as pd
from gymnasium.utils import seeding # type: ignore

# Action Space
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version8/src/workload2/2action.csv')
action_space = action_space_df.values.tolist()
action_space_copy = action_space_df.values.tolist()

# Convert to one-hot
action_mapping = {
    "ring": 1,
    "tree": 2,
    "random": 3,
    "a":1,
    "b":2
}

def convert_action_space(data, action_mapping):
    converted_data = []
    
    for line in data:
        # Convert action in column 3
        action2 = line[3]  
        if action2 in action_mapping:
            line[3] = action_mapping[action2]
        action3 = line[4]
        if action3 in action_mapping:
            line[4] = action_mapping[action3]
        
        # Convert action in column 2 (assuming it's a space-separated string of integers)
        action11 = line[2]
        action_list = list(map(int, action11.split(' ')))
        
        # Update the line with the transformed action list and action2 (now an integer)
        line = line[:2] + action_list + line[3:] 
        
        # Add the transformed line to the converted data list
        converted_data.append(line)
    
    return converted_data

converted_data = convert_action_space(action_space_copy, action_mapping)

state_df1 = pd.read_csv('/home/nlsde/RLmodel/Version8/src/workload2/2output_16.csv')
state_df2 = pd.read_csv('/home/nlsde/RLmodel/Version8/src/workload2/2output_17.csv')
state_df3 = pd.read_csv('/home/nlsde/RLmodel/Version8/src/workload2/2output_18.csv')
state_df4 = pd.read_csv('/home/nlsde/RLmodel/Version8/src/workload2/2output_20.csv')

state_space_df1 = state_df1.copy()
state_space_df2 = state_df2.copy()
state_space_df3 = state_df3.copy()
state_space_df4 = state_df4.copy()
#Deal with inf
state_space_df1.replace([np.inf, -np.inf], np.nan, inplace=True)
state_space_df2.replace([np.inf, -np.inf], np.nan, inplace=True)
state_space_df3.replace([np.inf, -np.inf], np.nan, inplace=True)
state_space_df4.replace([np.inf, -np.inf], np.nan, inplace=True)
#Deal with Nan
state_space_df1 = state_space_df1.fillna(state_df2)
state_space_df1 = state_space_df1.ffill()

for col in state_df2.columns:
    state_space_df2[col] = state_space_df2[col].fillna(state_space_df1[col])
    state_space_df3[col] = state_space_df3[col].fillna(state_space_df1[col])
    state_space_df4[col] = state_space_df4[col].fillna(state_space_df1[col])

def process_state_space(state_space_df1, state_space_df2, state_space_df3, state_space_df4):
    state_space = []
    state_space.append([state_space_df1, state_space_df2, state_space_df3, state_space_df4]) 
    return state_space

state_space = process_state_space(state_space_df1, state_space_df2, state_space_df3, state_space_df4)
# Define the reward, security, step, reset
MAX_TIMESTEPS = 500

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
        self.SEm = pow(action_space_df['number'].max(), 2.29)
        self.state = None
        self.g = 1 
        self.timeCounts = 0

    def reward(self, L_t, SE_t, TPS_t):
        r_t = 0.1 * np.exp(1 - L_t/self.Lm) + 0.1 * np.exp(SE_t/self.SEm) + 0.8 * np.exp(TPS_t/self.TPSm)
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

    def step(self, action,index):
        action = int(action)
        selected_action = action_space[action]
        period,gaslimit,transactionpool,connection,position,number = selected_action
        self.state = np.array(state_space[0][index].iloc[action])
        self.realstate = state_space_df1.iloc[action]

        L_t = self.realstate[5]
        SE_t = self.security(number, connection)
        TPS_t = self.realstate[2]

        current_reward = self.reward(L_t, SE_t, TPS_t)

        self.timeCounts += 1
        #if ((self.TPSm - TPS_t) / self.TPSm < 0.05) or self.timeCounts > MAX_TIMESTEPS: done=True
        #else: done=False
        done=False
        
        return self.state, current_reward, done, {}
    
    def reset(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        sample_action = self.np_random.integers(low=0, high=len(action_space)-1, size=1)[0] #action index
        new_state = []
        for i in range(4):
            space = state_space[0][i]
            new_state.append(space.iloc[sample_action])
        self.state = np.array(new_state)
        self.timeCounts = 0
        return self.state, {}

    def initial(self):
        sample_action = 1279
        self.state = [statespace[sample_action] for statespace in state_space]
        self.timeCounts = 0
        return self.state, sample_action