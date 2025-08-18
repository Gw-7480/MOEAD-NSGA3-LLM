import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义策略网络（Actor），用于输出连续卸载因子和离散动作（如卫星选择）
class HybridPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dims):
        super().__init__()
        # 连续动作参数（三个卸载因子 μ_m, μ_n, μ_l）
        self.continuous_action_dim = 3

        # 共享特征提取层：两层全连接+ReLU，用于提取状态特征
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 256),  # 输入层到256维
            nn.ReLU(),
            nn.Linear(256, 256),        # 256维到256维
            nn.ReLU()
        )

        # 连续动作分支：输出高斯分布的均值（μ）和对数标准差（log_std）
        self.continuous_mu = nn.Linear(256, self.continuous_action_dim)  # 输出均值
        self.continuous_log_std = nn.Parameter(torch.zeros(self.continuous_action_dim))  # 可学习的log_std参数

        # 离散动作分支：输出离散动作（如卫星选择）的logits
        self.discrete_logits = nn.Linear(256, action_dims['discrete'])

    def forward(self, state):
        # 通过共享特征层提取特征
        features = self.shared_net(state)

        # 连续动作分支：输出均值（sigmoid约束到[0,1]），标准差（exp保证正数）
        mu = torch.sigmoid(self.continuous_mu(features))
        std = torch.exp(self.continuous_log_std)

        # 离散动作分支：输出logits
        discrete_logits = self.discrete_logits(features)

        # 返回连续动作参数（均值、标准差）和离散动作logits
        return {'continuous': (mu, std), 'discrete': discrete_logits}


# 定义Q网络（Critic），用于评估状态-动作对的价值
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dims):
        super().__init__()
        # 总动作维度 = 连续动作维度 + 离散动作维度
        total_action_dim = action_dims['continuous'] + action_dims['discrete']
        # Q网络结构：输入为状态和动作拼接，输出为Q值
        self.net = nn.Sequential(
            nn.Linear(state_dim + total_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 输出单个Q值
        )

    def forward(self, state, actions):
        # 合并连续和离散动作为一个向量
        combined_actions = torch.cat([actions['continuous'], actions['discrete']], dim=-1)
        # 拼接状态和动作，输入Q网络
        return self.net(torch.cat([state, combined_actions], dim=-1))


# SAC智能体（简化版），包含策略网络和两个Q网络
class SACAgent:
    def __init__(self, state_dim, action_dims):
        # 策略网络（Actor）
        self.policy_net = HybridPolicyNetwork(state_dim, action_dims)
        # 两个Q网络（Critic）
        self.q_net1 = QNetwork(state_dim, action_dims)
        self.q_net2 = QNetwork(state_dim, action_dims)

        # 目标Q网络（用于稳定训练，参数软更新）
        self.target_q_net1 = QNetwork(state_dim, action_dims)
        self.target_q_net2 = QNetwork(state_dim, action_dims)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())  # 初始化为主Q网络参数
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.q_optimizer = optim.Adam(list(self.q_net1.parameters()) + list(self.q_net2.parameters()), lr=3e-4)

        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005   # 目标网络软更新率
        self.alpha = 0.2   # 熵系数（鼓励探索）

    def select_action(self, state):
        # 将输入状态转为Tensor，并增加batch维
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 前向传播，不计算梯度
        with torch.no_grad():
            policy_output = self.policy_net(state_tensor)

            # 连续动作采样（高斯分布采样后tanh约束到[-1,1]，再映射到[0,1]）
            mu, std = policy_output['continuous']
            continuous_actions = torch.tanh(mu + torch.randn_like(std) * std)
            continuous_actions = (continuous_actions + 1) / 2  # [-1,1] -> [0,1]

            # 离散动作采样（如卫星选择，softmax后多项分布采样）
            discrete_probs = torch.softmax(policy_output['discrete'], dim=-1)
            discrete_action = torch.multinomial(discrete_probs, 1).item()

        # 返回连续动作（卸载因子）和离散动作（卫星选择）
        return {
            'continuous': continuous_actions.squeeze().numpy(),  # [μ_m, μ_n, μ_l]
            'discrete': discrete_action
        }

    def update(self, batch):
        # SAC的训练更新逻辑（此处省略，需结合实际任务实现）
        pass


# 参数初始化示例
if __name__ == "__main__":
    # 状态维度（示例参数）
    state_dim = 64  # 论文中包括信道状态、剩余任务等
    action_dims = {
        'continuous': 3,  # 三个卸载因子 μ_m, μ_n, μ_l
        'discrete': 5  # 卫星选择（假设5个卫星）
    }

    # 初始化智能体
    agent = SACAgent(state_dim, action_dims)

    # 示例动作选择
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print("卸载因子:", action['continuous'])  # 例如 [0.3, 0.8, 0.1]
    # print("卫星选择:", action['discrete'])  # 例如 3