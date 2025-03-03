import torch as T
import torch.nn as nn
import torch.nn.functional as F

class WeightMLP(nn.Module):
    def __init__(self, input_size):
        super(WeightMLP, self).__init__()
        # 定义网络层
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, input_size)
        
    def forward(self, x):
        # 前向传播
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        
        # 使用softmax确保输出的权重和为1
        weights = F.softmax(x, dim=-1)
        return weights

class WeightNetwork:
    def __init__(self, input_size, learning_rate=0.001, temperature=1.0):
        """
        初始化权重网络
        Args:
            input_size: 输入维度
            learning_rate: 学习率
            temperature: 温度参数
        """
        # 设置设备
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 初始化模型并移至设备
        self.model = WeightMLP(input_size).to(self.device)
        self.optimizer = T.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.temperature = temperature
        
        # 初始化累积权重W为0
        self.accumulated_W = T.zeros(input_size)
        # update counter
        self.U = 0
    
    def update_W(self, r):
        """
        更新累积权重W
        Args:
            r: 当前输入的奖励值
        """
        # 对批次维度求平均，得到[input_size]维度的更新值
        r_mean = T.mean(r, dim=0)
        self.accumulated_W += r_mean
        self.U += 1

    def get_accumulated_W(self):
        """
        获取当前累积的W值
        """
        normalized_W = self.accumulated_W / (self.U + 1e-10)
        return normalized_W, self.U

    def compute_loss(self, predicted_weights, r):
        """
        计算损失函数
        """
        # 更新累积权重W并计算目标分布
        self.update_W(r.to(self.device))
        # 避免除零
        epsilon = 1e-10
        normalized_W = self.accumulated_W / (self.U + epsilon)
        
        # 为了数值稳定性，减去最大值
        scaled_W = self.temperature * normalized_W
        max_w = T.max(scaled_W)
        exp_weights = T.exp(scaled_W - max_w)
        denominator = exp_weights.sum()
        target_distribution = exp_weights / (denominator + epsilon)

        # 扩展target_distribution到批次维度
        target_distribution = target_distribution.unsqueeze(0).expand(predicted_weights.shape[0], -1)
        
        # 添加梯度裁剪以防止梯度爆炸
        T.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        loss = T.mean((predicted_weights - target_distribution) ** 2)
        # 检查loss是否为nan
        if T.isnan(loss):
            print("Warning: Loss is NaN!")
            print("Accumulated W:", self.accumulated_W)
            print("Target distribution:", target_distribution)
            print("Predicted weights:", predicted_weights)

        return loss
    
    def update(self, input_data, r):
        """
        更新网络参数
        """
        # 将输入数据移至GPU
        input_data = input_data.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        predicted_weights = self.model(input_data)
        
        # 计算损失
        loss = self.compute_loss(predicted_weights, r)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, input_data):
        """
        预测权重
        """
        input_data = input_data.to(self.device)
        
        self.model.eval()
        with T.no_grad():
            predictions = self.model(input_data)
            return predictions.cpu()
    
    def save_model(self, path):
        """
        保存模型
        """
        T.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """
        加载模型
        """
        self.model.load_state_dict(T.load(path)) 