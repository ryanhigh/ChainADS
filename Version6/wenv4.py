import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pandas as pd

# Read CSV files for action space and state data
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version6/src/workload2/2action.csv')
action_space = action_space_df.values.tolist()
state_df1 = pd.read_csv('/home/nlsde/RLmodel/Version6/src/workload2/2output_16.csv')
state_df2 = pd.read_csv('/home/nlsde/RLmodel/Version6/src/workload2/2output_17.csv')
state_df3 = pd.read_csv('/home/nlsde/RLmodel/Version6/src/workload2/2output_18.csv')
state_df4 = pd.read_csv('/home/nlsde/RLmodel/Version6/src/workload2/2output_20.csv')

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

MAX_TIMESTEPS = 600

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        
        # 动作空间的大小与CSV中的行数相同
        self.action_space = spaces.Discrete(len(action_space))
        self.observation_space = spaces.Box(
            low=state_space_df1.min().values,
            high=state_space_df1.max().values,
            dtype=np.float32
        )
        self.TPSm = state_df1['tps'].max()               # 这里在修改到MFC version时将state_space_df变为state_df1 
        self.Lm = state_df1['delay'].max()               # 这里在修改到MFC version时将state_space_df变为state_df1
        self.SEm = pow(action_space_df['number'].max(), 2.29)
        self.state = None
        self.timeCounts = 0
        # self.q = None   # 计算安全指标：表示网络规模q，由聚类系数计算

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

        se = pow(m, q)
        return se


    def step(self, action):
        # 根据action的索引获取CSV中的动作
        selected_action = action_space[action]
        period, gaslimit, transactionpool, connection, position, number = selected_action
        self.state = np.array(state_space[action])

        # 获取性能指标
        L_t = self.state[4]
        SE_t = self.security(number, connection)
        TPS_t = self.state[2]

        # 计算奖励函数reward
        current_reward = self.reward(L_t, SE_t, TPS_t)

        # 判断是否到达边界/训练结束
        self.timeCounts += 1


        if ((self.TPSm - TPS_t) / self.TPSm < 0.05) or self.timeCounts > MAX_TIMESTEPS: done=True
        else: 
            done=False
            # current_reward *= 0.8
        
        # 返回状态、奖励、是否完成、{}、其他信息
        return self.state, current_reward, done, TPS_t, L_t, SE_t
    
    def reset(self, seed=None):
        # sample_action =  self.action_space.sample() # 随机初始化一个环境状态
        self.np_random, seed = seeding.np_random(seed)
        sample_action = self.np_random.integers(low=0, high=len(action_space)-1, size=1)[0]
        # sample_action = 863
        self.state = np.array(state_space[sample_action])
        # self.state = np.array(state_space[self.np_random.uniform(low=0, high=len(action_space)-1)])
        self.timeCounts = 0
        return self.state, sample_action
    
    def initial(self):
        sample_action = 1279
        self.state = np.array(state_space[sample_action])
        self.timeCounts = 0
        return self.state, sample_action
    


# if __name__ == "__main__":
#     env = CustomEnv()
#     # for i in range(10):
#     #     observation = env.reset(seed=19)
#     #     observation = observation[0] 
#     #     print(observation)
#     num = [1, 2, 4]
#     graph = ['random', 'ring', 'tree']
#     se_max = 0
#     for number in num:
#         for connection in graph:
#             SE_t = env.security(number, connection)
#             if SE_t > se_max:
#                 se_max = SE_t
    
#     rwd = 0
#     for i in range(len(state_df1)):
#         df_row = state_df1.iloc[i]
#         L_t = df_row.iloc[5]
#         TPS_t = df_row.iloc[2]
#         ri = env.reward(L_t, se_max, TPS_t)
#         if ri > rwd:
#             rwd = ri
    
#     print(rwd)
