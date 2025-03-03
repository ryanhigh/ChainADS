import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class RandomForest():
    def __init__(self, state_df, action_df):
        self.state_df1 = state_df
        self.action_df = action_df
        self.drop_list = ['tps', 'delay']
        self.object_list = ['transactionpool', 'connection']
        self.action_features = None
        self.X = None
        self.y = None
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def get_performance(self, L_t, SE_t, TPS_t):
        TPSm = self.state_df1['tps'].max()                
        Lm = self.state_df1['delay'].max()              
        SEm = 1
        r_t = 0.4 * np.exp(1 - L_t/Lm) + 0.2 * np.exp(SE_t/SEm) + 0.4 * np.exp(TPS_t/TPSm)
        return r_t

    def security(self, m, connection):
        cluttering_coefficient_lt = [0.00, 0.29, 0.24]
        connect_lt = [1, 2, 0]
        for i in range(len(connect_lt)):
            if connect_lt[i] == connection:
                q = cluttering_coefficient_lt[i] + 2
                break

        se = pow(m, q)
        return se
    
    def concat(self):
        states_df = self.state_df1
        actions_df = self.action_df
        X = []
        y = []

        for state, action in zip(states_df.iterrows(), actions_df.iterrows()):
            # 获取当前行的状态和动作
            _, state_row = state
            _, action_row = action

            state_row_slim = state_row.drop(columns=self.drop_list)
            # 拼接当前状态和动作作为特征
            combined_features = np.concatenate([state_row_slim.values, action_row.values])
            X.append(combined_features)

            # 获取性能指标
            L_t = state_row['delay']
            SE_t = self.security(1, action_row['connection'])
            TPS_t = state_row['tps']

            # 模拟性能标签 (y) - 实际应用中需要根据真实数据替换
            performance = self.get_performance(L_t, SE_t, TPS_t)  # 这里随机生成性能数据
            y.append(performance)

        self.X = np.array(X)
        self.y= np.array(y)
    
    def preprocess(self):
         # 使用 LabelEncoder 将非数值特征转为数值
        encoder = LabelEncoder()
        for item in self.object_list:
            self.action_df[item] = encoder.fit_transform(self.action_df[item])
        
        self.action_features = self.action_df.values
        
    def first_state(self):
        random_index = np.random.randint(0, len(self.state_df1) - 1)
        current_state = self.state_df1.iloc[random_index].values  
        return current_state
    
    def predict(self, current_state):
        self.model.fit(self.X, self.y)

        # 6. 对当前状态的所有动作进行性能预测
        state_action_inputs = np.array([np.concatenate([current_state, action]) for action in self.action_features])
        predicted_performance = self.model.predict(state_action_inputs)

        # 7. 将预测性能与动作 ID 关联
        action_ids = self.action_df.index.values  # 动作 ID 使用行号
        action_performance = dict(zip(action_ids, predicted_performance))

        # 8. 选择最佳动作
        best_action_id = max(action_performance, key=action_performance.get)
        return best_action_id
    
    def prepare(self):
        self.preprocess()
        self.concat()

