import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from gymnasium.utils import seeding
from buffer import RolloutBuffer

# Read CSV files for action space and state data
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version11/src/workload2/2action.csv')
action_space = action_space_df.values.tolist()
state_df1 = pd.read_csv('/home/nlsde/RLmodel/Version11/src/workload2/2output_16.csv')
state_df2 = pd.read_csv('/home/nlsde/RLmodel/Version11/src/workload2/2output_17.csv')
state_df3 = pd.read_csv('/home/nlsde/RLmodel/Version11/src/workload2/2output_18.csv')
state_df4 = pd.read_csv('/home/nlsde/RLmodel/Version11/src/workload2/2output_20.csv')

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

MAX_TIMESTEPS = 600

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
        # Hyperparameter: ha(alpha), hb(beta), hc(gamma) = 0.33
        plus = 0
        r_t = 0.1 * np.exp(1 - L_t/self.Lm + plus) + 0.1 * np.exp(SE_t/self.SEm + plus) + 0.8 * np.exp(TPS_t/self.TPSm + plus)
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

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        for i in range(4):
            action = actions[i]
            selected_action = action_space[action]
            period,gaslimit,transactionpool,connection,position,number = selected_action
            #change
            state = state_space[i][action]
            # realstate
            realstate = state_space_df1.iloc[action]

            L_t = realstate[5]
            SE_t = self.security(number, connection)
            TPS_t = realstate[2]

            current_reward = self.reward(L_t, SE_t, TPS_t)
            rewards.append(current_reward)

            self.timeCounts += 1
            # if ((self.TPSm - TPS_t) / self.TPSm < 0.05) or self.timeCounts > MAX_TIMESTEPS: done=True
            # else: done=False
            done = False
            dones.append(done)

        max_reward = max(rewards)
        max_reward_index = rewards.index(max_reward)
        max_action = actions[max_reward_index]

        for i in range(4):
            state = state_space[i][max_action]
            states.append(state)
            self.buffers[i].states.append(state)
            self.buffers[i].actions.append(max_action)
            self.buffers[i].is_terminals.append(dones[i])
            self.buffers[i].rewards.append(max_reward)
            rewards.append(max_reward)
        return states, rewards, dones
    
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
