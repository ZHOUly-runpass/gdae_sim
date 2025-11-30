"""
经验回放缓冲区（Replay Buffer）
用于存储和采样训练经验
"""
import numpy as np
import torch


class ReplayBuffer:
    """
    经验回放缓冲区
    存储 (state, action, reward, done, next_state) 元组
    """

    def __init__(self, max_size=int(1e6), seed=0):
        """
        初始化回放缓冲区

        Args:
            max_size: 缓冲区最大容量
            seed: 随机种子
        """
        self.max_size = max_size
        self.ptr = 0  # 当前指针位置
        self.size = 0  # 当前缓冲区大小

        # 存储数据的数组（延迟初始化）
        self.state = None
        self.action = None
        self.reward = None
        self.done = None
        self.next_state = None

        # 设置随机种子
        np.random.seed(seed)

    def add(self, state, action, reward, done, next_state):
        """
        添加一条经验到缓冲区

        Args:
            state: 当前状态 [state_dim]
            action: 执行的动作 [action_dim]
            reward: 获得的奖励 (float)
            done: 是否结束 (bool)
            next_state: 下一个状态 [state_dim]
        """
        # 延迟初始化数组（第一次添加时确定维度）
        if self.state is None:
            state_dim = len(state) if isinstance(state, (list, np.ndarray)) else state.shape[0]
            action_dim = len(action) if isinstance(action, (list, np.ndarray)) else action.shape[0]

            self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
            self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
            self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
            self.done = np.zeros((self.max_size, 1), dtype=np.float32)
            self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)

        # 存储数据
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.next_state[self.ptr] = next_state

        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        """
        从缓冲区随机采样一批数据

        Args:
            batch_size: 批次大小

        Returns:
            tuple: (states, actions, rewards, dones, next_states)
                   所有数组形状为 [batch_size, ...]
        """
        # 随机采样索引
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
            self.next_state[indices]
        )

    def __len__(self):
        """返回当前缓冲区大小"""
        return self.size

    def is_ready(self, batch_size):
        """检查缓冲区是否有足够的数据进行采样"""
        return self.size >= batch_size

    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

    def save(self, filename):
        """保存缓冲区到文件"""
        np.savez(
            filename,
            state=self.state[:self.size],
            action=self.action[:self.size],
            reward=self.reward[:self.size],
            done=self.done[:self.size],
            next_state=self.next_state[:self.size],
            ptr=self.ptr,
            size=self.size
        )
        print(f"Replay buffer saved to {filename}")

    def load(self, filename):
        """从文件加载缓冲区"""
        data = np.load(filename)

        self.state = data['state']
        self.action = data['action']
        self.reward = data['reward']
        self.done = data['done']
        self.next_state = data['next_state']
        self.ptr = int(data['ptr'])
        self.size = int(data['size'])

        print(f"Replay buffer loaded from {filename}, size={self.size}")


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Replay Buffer")
    print("=" * 60)

    # 创建缓冲区
    buffer = ReplayBuffer(max_size=1000, seed=42)
    print(f"\n✓ 创建缓冲区: max_size=1000")

    # 添加数据
    print("\n" + "=" * 60)
    print("添加经验测试")
    print("=" * 60)

    state_dim = 24
    action_dim = 2
    num_samples = 100

    for i in range(num_samples):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        done = np.random.rand() > 0.9
        next_state = np.random.randn(state_dim)

        buffer.add(state, action, reward, done, next_state)

    print(f"✓ 添加 {num_samples} 条经验")
    print(f"  当前缓冲区大小: {len(buffer)}")
    print(f"  指针位置: {buffer.ptr}")

    # 采样测试
    print("\n" + "=" * 60)
    print("采样测试")
    print("=" * 60)

    batch_size = 32
    states, actions, rewards, dones, next_states = buffer.sample_batch(batch_size)

    print(f"✓ 采样批次大小: {batch_size}")
    print(f"  states 形状: {states.shape}")
    print(f"  actions 形状: {actions.shape}")
    print(f"  rewards 形状: {rewards.shape}")
    print(f"  dones 形状: {dones.shape}")
    print(f"  next_states 形状: {next_states.shape}")

    assert states.shape == (batch_size, state_dim), "states 形状错误"
    assert actions.shape == (batch_size, action_dim), "actions 形状错误"
    assert rewards.shape == (batch_size, 1), "rewards 形状错误"
    assert dones.shape == (batch_size, 1), "dones 形状错误"
    assert next_states.shape == (batch_size, state_dim), "next_states 形状错误"

    print("✓ 所有形状检查通过")

    # 缓冲区满载测试
    print("\n" + "=" * 60)
    print("缓冲区满载测试")
    print("=" * 60)

    for i in range(1500):  # 超过 max_size
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        done = False
        next_state = np.random.randn(state_dim)
        buffer.add(state, action, reward, done, next_state)

    print(f"✓ 添加 1500 条经验（超过容量）")
    print(f"  当前缓冲区大小: {len(buffer)} (应该等于 max_size=1000)")
    print(f"  指针位置: {buffer.ptr}")

    assert len(buffer) == 1000, "缓冲区大小应该被限制在 max_size"
    print("✓ 缓冲区容量限制正常")

    # 保存和加载测试
    print("\n" + "=" * 60)
    print("保存和加载测试")
    print("=" * 60)

    import tempfile
    import os

    temp_file = os.path.join(tempfile.gettempdir(), "test_buffer.npz")

    buffer.save(temp_file)
    print(f"✓ 缓冲区已保存")

    # 创建新缓冲区并加载
    new_buffer = ReplayBuffer(max_size=1000)
    new_buffer.load(temp_file)

    print(f"✓ 缓冲区已加载")
    print(f"  加载后大小: {len(new_buffer)}")

    assert len(new_buffer) == len(buffer), "加载后大小不一致"

    # 清理临时文件
    os.remove(temp_file)

    print("\n" + "=" * 60)
    print("✓ 所有 Replay Buffer 测试通过！")
    print("=" * 60)