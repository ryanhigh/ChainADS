import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CrossLayer, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)
        self.b = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        x = x + self.W(x) + self.b
        return x
    
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        # 用 CrossLayer 替代直接的参数列表
        self.layers = nn.ModuleList([CrossLayer(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)  # 调用每一层的 forward 方法
        return x

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim):
        super(DeepNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x): #x=[x1,x2,x3,x4]
        h_prev = 1
        re = []
        for idx, layer in enumerate(self.hidden_layers):
            current_input = x[idx]
            h = layer(current_input)
            h = F.relu(h)
            h_current = h_prev * h
            h_prev = h_current
            re.append(h_prev)
        output = torch.cat(re, dim=-1)
        return output

class HybridModel(nn.Module):
    def __init__(self, cross_dim, deep_dim, num_cross_layers, num_deep_layers, hidden_dim):
        super(HybridModel, self).__init__()
        # System-level feature processing using Cross Network
        self.cross_network = CrossNetwork(cross_dim, num_cross_layers)
        # Node-level feature processing using Deep Network
        self.deep_network = DeepNetwork(deep_dim, num_deep_layers, hidden_dim)

    def forward(self, x_system, x_node):
        # Forward pass through the Cross Network
        cross_output = self.cross_network(x_system)
        
        # Forward pass through the Deep Network
        deep_output = self.deep_network(x_node)

        # Combine outputs of Cross Network and Deep Network
        combined_output = torch.cat((cross_output, deep_output), dim=-1)  # Concatenate along feature axis
        return combined_output
