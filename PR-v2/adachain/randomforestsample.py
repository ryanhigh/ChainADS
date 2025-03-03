import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
import cupy as cp

class AdaChainLearningAgent:
    def __init__(self, n_actions, n_estimators=100, max_depth=10):
        """
        初始化学习代理
        :param n_actions: 动作空间的大小
        :param n_estimators: 随机森林中的树的数量
        :param max_depth: 每棵树的最大深度
        """
        self.n_actions = n_actions
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        self.experience_buffer = []  # 用于存储 (state, action, reward) 的经验
        self.is_trained = False      # 标记模型是否已训练

    def add_experience(self, state, action, reward):
        """
        将经验添加到缓冲区
        :param state: 当前状态 (numpy array)
        :param action: 采取的动作 (int)
        :param reward: 获得的奖励 (float)
        """
        self.experience_buffer.append((state, action, reward))

    def train_model(self):
        """
        用经验缓冲区中的数据训练随机森林
        """
        if not self.experience_buffer:
            print("经验缓冲区为空，无法训练模型。")
            return
        
        # 构造训练数据
        X = []
        y = []
        for state, action, reward in self.experience_buffer:
            X.append(np.concatenate((state, [action])))  # 拼接状态和动作
            y.append(reward)
        
        # 自举法 (Bootstrap) 训练
        X, y = np.array(X), np.array(y)
        X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X))
        self.model.fit(X_resampled, y_resampled)
        self.is_trained = True

    def predict_performance(self, state):
        """
        预测在当前状态下不同动作的性能
        :param state: 当前状态 (numpy array)
        :return: 各动作的预测性能 (numpy array)
        """
        if not self.is_trained:
            print("模型尚未训练，返回随机性能预测。")
            return np.random.random(self.n_actions)
        
        # 构造输入数据，预测每个动作的性能
        predictions = []
        for action in range(self.n_actions):
            input_data = np.concatenate((state, [action]))
            predictions.append(self.model.predict([input_data])[0])
        return np.array(predictions)

    def select_action(self, state):
        """
        基于预测选择最优动作
        :param state: 当前状态 (numpy array)
        :return: 选择的动作 (int)
        """
        predicted_performance = self.predict_performance(state)
        # 如果多个动作预测性能相同，随机选择一个
        best_actions = np.argwhere(predicted_performance == np.max(predicted_performance)).flatten()
        return np.random.choice(best_actions)

# # 示例运行
# if __name__ == "__main__":
#     # 假设状态空间维度为4，动作空间大小为5
#     agent = AdaChainLearningAgent(n_actions=5)

#     # 模拟经验数据
#     for _ in range(100):
#         state = np.random.random(4)  # 随机生成状态
#         action = np.random.randint(0, 5)  # 随机动作
#         reward = np.random.random()  # 随机奖励
#         agent.add_experience(state, action, reward)

#     # 训练模型
#     agent.train_model()

#     # 模拟当前状态，选择动作
#     current_state = np.random.random(4)
#     selected_action = agent.select_action(current_state)
#     print(f"在状态 {current_state} 下选择的动作是 {selected_action}")
