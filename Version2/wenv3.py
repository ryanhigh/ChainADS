import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pandas as pd
from MFC import MFC

# 读取CSV文件
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src/small_action3.csv')
action_space = action_space_df.values.tolist()

# # 负载1
# state_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src/small_state.csv')
# state_space_df = state_space_df.fillna(state_space_df.shift())
# state_space = state_space_df.values.tolist()

# # 负载2
# state_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src/w3_res_12_processed.csv')
# state_space_df = state_space_df.fillna(state_space_df.shift())
# if state_space_df.isnull().values.any():
#     state_space_df = state_space_df.fillna(state_space_df.shift())
# state_space = state_space_df.values.tolist()
# state_df1 = state_space_df

# 负载3
state_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src/w4_res_fine_state.csv')
state_space_df = state_space_df.fillna(state_space_df.shift())
if state_space_df.isnull().values.any():
    state_space_df = state_space_df.fillna(state_space_df.shift())
state_space = state_space_df.values.tolist()
state_df1 = state_space_df

# # 负载B
# state_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src2/workloadB_scaled.csv')
# state_space = state_space_df.values.tolist()
# state_df1 = state_space_df

# # 负载B
# state_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src2/workloadB_with_nan.csv')
# state_space_df = state_space_df.fillna(state_space_df.shift())
# if state_space_df.isnull().values.any():
#     state_space_df = state_space_df.fillna(state_space_df.shift())
# state_space = state_space_df.values.tolist()
# state_df1 = state_space_df

MAX_TIMESTEPS = 100

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        
        # 动作空间的大小与CSV中的行数相同
        self.action_space = spaces.Discrete(len(action_space))
        self.observation_space = spaces.Box(
            low=state_space_df.min().values,
            high=state_space_df.max().values,
            dtype=np.float32
        )
        self.TPSm = state_df1['tps'].max()               # 这里在修改到MFC version时将state_space_df变为state_df1 
        self.Lm = state_df1['delay'].max()               # 这里在修改到MFC version时将state_space_df变为state_df1
        # self.SEm = pow(action_space_df['number'].max(), 0.29)
        self.SEm = 1
        self.state = None
        self.timeCounts = 0
        # self.q = None   # 计算安全指标：表示网络规模q，由聚类系数计算

    def reward(self, L_t, SE_t, TPS_t):
        # Hyperparameter: ha(alpha), hb(beta), hc(gamma) = 0.33
        r_t = 0.4 * np.exp(1 - L_t/self.Lm) + 0.2 * np.exp(SE_t/self.SEm) + 0.4 * np.exp(TPS_t/self.TPSm)
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
        period, gaslimit, transactionpool, connection, position = selected_action
        self.state = np.array(state_space[action])
        
        # # 这里可以根据你的逻辑处理这些参数
        # print(f"执行动作 - Period: {period}, GasLimit: {gaslimit}, "
        #       f"TransactionPool: {transactionpool}, Connection: {connection}, Position: {position}")
        # print("环境状态：", self.state)

        # 获取性能指标
        L_t = self.state[5]
        SE_t = self.security(2, connection)
        TPS_t = self.state[2]

        # print(f'延迟:{L_t}, 安全性:{SE_t}, TPS:{TPS_t}')
        # print("延迟与吞吐量最大值", self.Lm, self.TPSm)
        # 计算奖励函数reward
        current_reward = self.reward(L_t, SE_t, TPS_t)

        # 判断是否到达边界/训练结束
        self.timeCounts += 1
        if ((self.TPSm - TPS_t) / self.TPSm < 0.05) or self.timeCounts > MAX_TIMESTEPS: done=True
        else: done=False
        
        # 返回状态、奖励、是否完成、{}、其他信息
        return self.state, current_reward, done, TPS_t, L_t, SE_t
    
    def reset(self, seed=None):
        # sample_action =  self.action_space.sample() # 随机初始化一个环境状态
        self.np_random, seed = seeding.np_random(seed)
        sample_action = self.np_random.integers(low=0, high=len(action_space)-1, size=1)[0]
        self.state = np.array(state_space[sample_action])
        # self.state = np.array(state_space[self.np_random.uniform(low=0, high=len(action_space)-1)])
        self.timeCounts = 0
        return self.state, {}
    


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
