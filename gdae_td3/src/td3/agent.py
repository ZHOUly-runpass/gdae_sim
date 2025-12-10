"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 智能体
严格按照原项目训练逻辑实现
"""
import numpy as np
import torch
import torch.nn.functional as F
import copy
import os

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


class TD3Agent:
    """
    TD3 强化学习智能体
    """

    def __init__(
            self,
            state_dim=24,
            action_dim=2,
            max_action=1.0,
            device=None,
            # actor_lr=1e-3,
            # critic_lr=1e-3,
            actor_lr =  3e-4,  # 从 1e-3 降低
            critic_lr =  3e-4,  # 从 1e-3 降低
            gamma=0.99,
            tau=0.005,
            policy_noise=0.1,  # 0.2
            noise_clip=0.3,    # 0.5
            policy_freq=2
    ):
        """
        初始化 TD3 智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 动作最大值
            device: 计算设备
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子
            tau: 软更新系数
            policy_noise: 策略平滑噪声
            noise_clip: 噪声裁剪范围
            policy_freq: 策略更新频率
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # 创建 Actor 网络（当前网络和目标网络）
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # 创建 Critic 网络（当前网络和目标网络）
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 训练步数计数器
        self.total_it = 0

    def get_action(self, state, add_noise=False, noise_scale=0.1):
        """
        根据当前状态选择动作

        Args:
            state: 当前状态 [state_dim] 或 [batch_size, state_dim]
            add_noise: 是否添加探索噪声
            noise_scale: 噪声缩放系数

        Returns:
            action: [action_dim] 或 [batch_size, action_dim]
        """
        # 转换为 tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        # 如果是单个状态，添加批次维度
        if state.ndim == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # 获取动作
        with torch.no_grad():
            action = self.actor(state)

        # 添加探索噪声
        if add_noise:
            noise = torch.randn_like(action) * noise_scale
            action = (action + noise).clamp(-self.max_action, self.max_action)

        # 转换为 numpy
        action = action.cpu().numpy()

        # 移除批次维度（如果原始输入是单个状态）
        if squeeze_output:
            action = action[0]

        return action

    def train(
            self,
            replay_buffer,
            batch_size=256,
            discount=None,
            tau=None,
            policy_noise=None,
            noise_clip=None,
            policy_freq=None
    ):
        """
        训练 TD3 网络

        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            discount: 折扣因子（可选，默认使用初始化值）
            tau: 软更新系数（可选）
            policy_noise: 策略噪声（可选）
            noise_clip: 噪声裁剪（可选）
            policy_freq: 策略更新频率（可选）

        Returns:
            dict: 训练统计信息
        """
        self.total_it += 1

        # 使用默认参数
        discount = discount if discount is not None else self.gamma
        tau = tau if tau is not None else self.tau
        policy_noise = policy_noise if policy_noise is not None else self.policy_noise
        noise_clip = noise_clip if noise_clip is not None else self.noise_clip
        policy_freq = policy_freq if policy_freq is not None else self.policy_freq

        # 从 replay buffer 采样
        state, action, reward, done, next_state = replay_buffer.sample_batch(batch_size)

        # 转换为 tensor
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)

        with torch.no_grad():
            # 计算目标 Q 值
            # 使用目标 Actor 网络选择下一个动作
            next_action = self.actor_target(next_state)

            # 添加策略平滑噪声（Target Policy Smoothing）
            noise = (torch.randn_like(next_action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # 计算目标 Q 值（使用双 Q 网络的最小值）
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # 计算目标值：r + γ * (1 - done) * Q(s', a')
            target_Q = reward + (1 - done) * discount * target_Q

        # 获取当前 Q 值估计
        current_Q1, current_Q2 = self.critic(state, action)

        # 计算 Critic 损失（MSE）
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟策略更新（Delayed Policy Updates）
        actor_loss = None
        if self.total_it % policy_freq == 0:
            # 计算 Actor 损失（最大化 Q 值）
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # 优化 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self._soft_update(self.actor, self.actor_target, tau)
            self._soft_update(self.critic, self.critic_target, tau)

        # 返回训练统计
        stats = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
            'q1_value': current_Q1.mean().item(),
            'q2_value': current_Q2.mean().item(),
            'target_q_value': target_Q.mean().item()
        }

        return stats

    def _soft_update(self, source, target, tau):
        """
        软更新目标网络
        θ_target = τ * θ_source + (1 - τ) * θ_target

        Args:
            source: 源网络
            target: 目标网络
            tau: 软更新系数
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        """
        保存模型

        Args:
            filename: 保存路径（不含扩展名）
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it
        }, f"{filename}.pth")

        print(f"Model saved to {filename}. pth")

    def load(self, filename):
        """
        加载模型

        Args:
            filename: 加载路径（不含扩展名）
        """
        checkpoint = torch.load(f"{filename}.pth", map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']

        print(f"Model loaded from {filename}.pth (iteration {self.total_it})")


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 TD3 Agent")
    print("=" * 60)

    # 创建智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    agent = TD3Agent(
        state_dim=24,
        action_dim=2,
        max_action=1.0,
        device=device
    )

    print("✓ TD3 Agent 创建成功")

    # 测试动作选择
    print("\n" + "=" * 60)
    print("动作选择测试")
    print("=" * 60)

    test_state = np.random.randn(24)
    action = agent.get_action(test_state, add_noise=False)

    print(f"输入状态形状: {test_state.shape}")
    print(f"输出动作形状: {action.shape}")
    print(f"动作值: {action}")
    print(f"动作范围: [{action.min():.3f}, {action.max():.3f}]")

    assert action.shape == (2,), "动作形状错误"
    print("✓ 动作选择测试通过")

    # 测试训练
    print("\n" + "=" * 60)
    print("训练功能测试")
    print("=" * 60)

    # 创建 replay buffer 并填充数据
    buffer = ReplayBuffer(max_size=10000)

    for _ in range(1000):
        state = np.random.randn(24)
        action = np.random.randn(2)
        reward = np.random.randn()
        done = np.random.rand() > 0.9
        next_state = np.random.randn(24)
        buffer.add(state, action, reward, done, next_state)

    print(f"✓ Replay buffer 填充完成，大小: {len(buffer)}")

    # 执行训练
    stats = agent.train(buffer, batch_size=64)

    print(f"\n训练统计:")
    print(f"  Critic Loss: {stats['critic_loss']:.4f}")
    print(f"  Actor Loss: {stats['actor_loss']}")
    print(f"  Q1 Value: {stats['q1_value']:.4f}")
    print(f"  Q2 Value: {stats['q2_value']:.4f}")
    print(f"  Target Q Value: {stats['target_q_value']:.4f}")

    print("✓ 训练功能测试通过")

    # 测试保存和加载
    print("\n" + "=" * 60)
    print("保存和加载测试")
    print("=" * 60)

    import tempfile

    temp_file = os.path.join(tempfile.gettempdir(), "test_td3_model")

    agent.save(temp_file)
    print("✓ 模型已保存")

    # 创建新 agent 并加载
    new_agent = TD3Agent(state_dim=24, action_dim=2, device=device)
    new_agent.load(temp_file)

    print("✓ 模型已加载")

    # 验证加载后的输出一致
    action1 = agent.get_action(test_state)
    action2 = new_agent.get_action(test_state)

    assert np.allclose(action1, action2), "加载后输出不一致"
    print("✓ 加载后输出一致")

    # 清理临时文件
    os.remove(f"{temp_file}.pth")

    print("\n" + "=" * 60)
    print("✓ 所有 TD3 Agent 测试通过！")
    print("=" * 60)