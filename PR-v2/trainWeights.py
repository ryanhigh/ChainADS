import torch
from WeightNet import WeightNetwork
import matplotlib.pyplot as plt
import pickle

def rns_reward(fname, mode, data):
    if mode=='save':
        # 保存列表到文件
        # pkl文件
        with open(fname, 'wb') as file:
            pickle.dump(data, file)
    elif mode=='read':
        # 从文件中读取列表
        with open(fname, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data
    

Sl = rns_reward('/home/nlsde/RLmodel/PR-v2/stateData.pkl', 'read', None)
R1 = rns_reward('/home/nlsde/RLmodel/PR-v2/stateReward1.pkl', 'read', None)
R2 = rns_reward('/home/nlsde/RLmodel/PR-v2/stateReward2.pkl', 'read', None)
print(f"R1:{R1}\nR2:{R2}\nSl:{Sl}")

def prepare_data(r1, r2):
    """
    准备训练数据
    Args:
        r1: 第一个奖励列表
        r2: 第二个奖励列表
    Returns:
        训练数据和目标权重
    """
    # 确保两个列表长度相同
    assert len(r1) == len(r2), "两个列表长度必须相同"
    
    # 将列表转换为tensor
    input_data = torch.tensor([[r1[i], r2[i]] for i in range(len(r1))], dtype=torch.float32)
    rewards = input_data.clone()  # 这里使用输入作为目标权重
    
    return input_data, rewards

def train(r1, r2, num_epochs=200, batch_size=32, learning_rate=0.0001):
    """
    训练权重网络
    Args:
        r1: 第一个奖励列表
        r2: 第二个奖励列表
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 平均损失历史池
    avg_losses = []

    # 准备数据
    input_data, rewards = prepare_data(r1, r2)
    dataset_size = len(input_data)
    
    # 初始化网络（输入维度为2）
    network = WeightNetwork(input_size=2, learning_rate=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        # 随机打乱数据顺序
        indices = torch.randperm(dataset_size)
        
        # 批次训练
        for i in range(0, dataset_size, batch_size):
            # 获取当前批次的索引
            batch_indices = indices[i:min(i + batch_size, dataset_size)]
            
            # 准备批次数据
            batch_input = input_data[batch_indices]
            batch_rewards = rewards[batch_indices]
            
            # 更新网络
            loss = network.update(batch_input, batch_rewards)
            total_loss += loss
        
        # 计算平均损失
        avg_loss = total_loss * batch_size / dataset_size
        
        # 打印训练信息
        avg_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            # 打印当前累积的W值
            W, U = network.get_accumulated_W()
            print(f'Accumulated W: {W}, Update Counter: {U}')
    
    return network, avg_losses

def plot_loss_curve(loss_history, save_path=None):
    """
    绘制损失曲线
    Args:
        loss_history: 损失历史记录
        save_path: 图像保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Loss Curve During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def evaluate(network, r1_test, r2_test):
    """
    评估网络性能
    Args:
        network: 训练好的网络
        r1_test: 测试数据r1
        r2_test: 测试数据r2
    """
    # 准备测试数据
    test_input = torch.tensor([[r1_test[i], r2_test[i]] 
                              for i in range(len(r1_test))], dtype=torch.float32)
    
    network.model.eval()
    with torch.no_grad():
        predicted_weights = network.predict(test_input)
        print("\n测试结果:")
        print("输入:")
        for i in range(len(r1_test)):
            print(f"r1: {r1_test[i]:.4f}, r2: {r2_test[i]:.4f}")
        print("\n预测权重:")
        print(predicted_weights)
        print("\n权重和:", predicted_weights.sum(dim=1))

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 示例数据
    r1 = R1 # 第一个奖励列表
    r2 = R2  # 第二个奖励列表
    
    # 训练参数
    num_epochs = 100
    batch_size = 2
    learning_rate = 0.001
    
    # 训练网络
    network, avg_loss = train(r1, r2, num_epochs, batch_size, learning_rate)
    
    # 绘制loss曲线
    plot_loss_curve(avg_loss, save_path='/home/nlsde/RLmodel/PR-v2/loss_curve.png')

    # 保存模型
    network.save_model('weight_model.pth')

    
    # 测试网络
    r1_test = [R1[38]]  # 测试数据
    r2_test = [R2[38]]
    print(f"test data: R1 give {r1_test}, R2 give {r2_test}")
    evaluate(network, r1_test, r2_test)

if __name__ == "__main__":
    main()