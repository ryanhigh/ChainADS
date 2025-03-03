import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pandas as pd
from MFC import MFC

# 读取CSV文件
action_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src2/actionB.csv')
action_space = action_space_df.values.tolist()

# # 负载B_train
# state_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src2/workloadB_scaled.csv')
# state_space = state_space_df.values.tolist()
# state_df1 = state_space_df

# 负载B_validate
state_space_df = pd.read_csv('/home/nlsde/RLmodel/Version2/src2/workloadB.csv')
state_space = state_space_df.values.tolist()
state_df1 = state_space_df
full_fp = '/home/nlsde/RLmodel/Version2/src2/B_matched.csv'

MAX_TIMESTEPS = 500

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
        self.SEm = pow(action_space_df['number'].max(), 2.29)
        self.state = None
        self.timeCounts = 0
        self.rmax = 0
        self.flag = 0                                   # 看是否是超时的结果
        # self.q = None   # 计算安全指标：表示网络规模q，由聚类系数计算

    def reward(self, L_t, SE_t, TPS_t):
        # Hyperparameter: ha(alpha), hb(beta), hc(gamma) = 0.33
        plus = 0
        r_t = 0.4 * np.exp(1 - L_t/self.Lm+plus) + 0.2 * np.exp(SE_t/self.SEm+plus) + 0.4 * np.exp(TPS_t/self.TPSm+plus)
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
    
    def Set_MaxReward(self, fp):
        data  =pd.read_csv(fp)
        rl = []
        for i in range(len(data)):
            row = data.iloc[i]
            number = row.iloc[5]
            connection = row.iloc[3]
            tps = row.iloc[8]
            delay = row.iloc[11]
            
            se = self.security(number, connection)
            r = self.reward(delay, se, tps)
            rl.append(r)

        R = {'reward':rl}
        df1 = pd.DataFrame(R)
        sortcounts = df1['reward'].sort_values(ascending=True)
        sorl = sortcounts.tolist()
        r_max = sorl[len(sorl)-1]
        self.rmax = r_max

    def step(self, action):
        # 根据action的索引获取CSV中的动作
        selected_action = action_space[action]
        period, gaslimit, transactionpool, connection, position, number = selected_action
        self.state = np.array(state_space[action])
        
        # # 这里可以根据你的逻辑处理这些参数
        # print(f"执行动作 - Period: {period}, GasLimit: {gaslimit}, "
        #       f"TransactionPool: {transactionpool}, Connection: {connection}, Position: {position}")
        # print("环境状态：", self.state)

        # 获取性能指标
        L_t = self.state[5]
        SE_t = self.security(number, connection)
        TPS_t = self.state[2]

        # 计算奖励函数reward
        current_reward = self.reward(L_t, SE_t, TPS_t)

        # 判断是否到达边界/训练结束
        self.timeCounts += 1
        if ((self.rmax - current_reward) / self.rmax < 0.05) or self.timeCounts >= MAX_TIMESTEPS:
            done = True
        # elif self.timeCounts > MAX_TIMESTEPS: 
        #     done = True
        #     self.flag = 1
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

        self.Set_MaxReward(full_fp)
        self.flag = 0
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
