"""
TD3 算法的 Actor 和 Critic 网络
严格按照原项目结构实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor 网络（策略网络）
    输入：state [batch_size, state_dim]
    输出：action [batch_size, action_dim]，范围 [-1, 1]
    """

    def __init__(self, state_dim=24, action_dim=2):
        """
        初始化 Actor 网络

        Args:
            state_dim: 状态维度（20 激光 + 2 目标信息 + 2 上次动作）
            action_dim: 动作维度（线速度、角速度）
        """
        super(Actor, self).__init__()

        # 网络结构：与原项目完全一致
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)

        # Tanh 激活函数，确保输出在 [-1, 1]
        self.tanh = nn.Tanh()

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.uniform_(self.layer_3.weight, -3e-3, 3e-3)

        nn.init.zeros_(self.layer_1.bias)
        nn.init.zeros_(self.layer_2.bias)
        nn.init.zeros_(self.layer_3.bias)

    def forward(self, state):
        """
        前向传播

        Args:
            state: [batch_size, state_dim] 或 [state_dim]

        Returns:
            action: [batch_size, action_dim] 或 [action_dim]，范围 [-1, 1]
        """
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        action = self.tanh(self.layer_3(x))

        return action


class Critic(nn.Module):
    """
    Critic 网络（双 Q 网络）
    输入：state + action
    输出：两个 Q 值估计
    """

    def __init__(self, state_dim=24, action_dim=2):
        """
        初始化 Critic 网络

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
        """
        super(Critic, self).__init__()

        # Q1 网络
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)  # 状态分支
        self.layer_2_a = nn.Linear(action_dim, 600)  # 动作分支
        self.layer_3 = nn.Linear(600, 1)

        # Q2 网络（独立的第二个 Q 网络）
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        # Q1 网络
        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.xavier_uniform_(self.layer_2_s.weight)
        nn.init.xavier_uniform_(self.layer_2_a.weight)
        nn.init.uniform_(self.layer_3.weight, -3e-3, 3e-3)

        # Q2 网络
        nn.init.xavier_uniform_(self.layer_4.weight)
        nn.init.xavier_uniform_(self.layer_5_s.weight)
        nn.init.xavier_uniform_(self.layer_5_a.weight)
        nn.init.uniform_(self.layer_6.weight, -3e-3, 3e-3)

        # 偏置初始化为 0
        for layer in [self.layer_1, self.layer_2_s, self.layer_2_a, self.layer_3,
                      self.layer_4, self.layer_5_s, self.layer_5_a, self.layer_6]:
            nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        """
        前向传播，计算两个 Q 值

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]

        Returns:
            q1: [batch_size, 1] 第一个 Q 值
            q2: [batch_size, 1] 第二个 Q 值
        """
        # Q1 网络计算
        s1 = F.relu(self.layer_1(state))

        # 状态和动作分别编码后融合（与原项目一致）
        s1_encoded = torch.mm(s1, self.layer_2_s.weight.data.t())
        a1_encoded = torch.mm(action, self.layer_2_a.weight.data.t())
        s1_fused = F.relu(s1_encoded + a1_encoded + self.layer_2_a.bias.data)

        q1 = self.layer_3(s1_fused)

        # Q2 网络计算（并行）
        s2 = F.relu(self.layer_4(state))

        s2_encoded = torch.mm(s2, self.layer_5_s.weight.data.t())
        a2_encoded = torch.mm(action, self.layer_5_a.weight.data.t())
        s2_fused = F.relu(s2_encoded + a2_encoded + self.layer_5_a.bias.data)

        q2 = self.layer_6(s2_fused)

        return q1, q2

    def Q1(self, state, action):
        """
        只计算 Q1 值（用于 Actor 更新）

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]

        Returns:
            q1: [batch_size, 1]
        """
        s1 = F.relu(self.layer_1(state))
        s1_encoded = torch.mm(s1, self.layer_2_s.weight.data.t())
        a1_encoded = torch.mm(action, self.layer_2_a.weight.data.t())
        s1_fused = F.relu(s1_encoded + a1_encoded + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1_fused)

        return q1


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 TD3 网络结构")
    print("=" * 60)

    # 创建网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    actor = Actor(state_dim=24, action_dim=2).to(device)
    critic = Critic(state_dim=24, action_dim=2).to(device)

    # 测试 Actor
    print("\n" + "=" * 60)
    print("Actor 网络测试")
    print("=" * 60)

    batch_size = 32
    test_state = torch.randn(batch_size, 24).to(device)
    test_action = actor(test_state)

    print(f"输入状态形状: {test_state.shape}")
    print(f"输出动作形状: {test_action.shape}")
    print(f"动作范围: [{test_action.min().item():. 3f}, {test_action.max().item():.3f}]")

    assert test_action.shape == (batch_size, 2), "Actor 输出形状错误"
    assert test_action.min() >= -1.0 and test_action.max() <= 1.0, "Actor 输出超出范围"
    print("✓ Actor 网络测试通过")

    # 测试 Critic
    print("\n" + "=" * 60)
    print("Critic 网络测试")
    print("=" * 60)

    test_q1, test_q2 = critic(test_state, test_action)

    print(f"输入状态形状: {test_state.shape}")
    print(f"输入动作形状: {test_action.shape}")
    print(f"Q1 输出形状: {test_q1.shape}")
    print(f"Q2 输出形状: {test_q2.shape}")
    print(f"Q1 值范围: [{test_q1.min().item():.3f}, {test_q1.max().item():.3f}]")
    print(f"Q2 值范围: [{test_q2.min().item():.3f}, {test_q2.max().item():.3f}]")

    assert test_q1.shape == (batch_size, 1), "Q1 输出形状错误"
    assert test_q2.shape == (batch_size, 1), "Q2 输出形状错误"
    print("✓ Critic 网络测试通过")

    # 测试单个 Q1 计算
    print("\n" + "=" * 60)
    print("单 Q1 计算测试")
    print("=" * 60)

    test_q1_only = critic.Q1(test_state, test_action)
    print(f"Q1 输出形状: {test_q1_only.shape}")
    print("✓ 单 Q1 计算测试通过")

    # 打印网络参数统计
    print("\n" + "=" * 60)
    print("网络参数统计")
    print("=" * 60)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())

    print(f"Actor 参数量: {actor_params:,}")
    print(f"Critic 参数量: {critic_params:,}")
    print(f"总参数量: {actor_params + critic_params:,}")

    print("\n" + "=" * 60)
    print("✓ 所有网络测试通过！")
    print("=" * 60)