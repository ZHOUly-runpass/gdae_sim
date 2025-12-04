"""
诊断动作转换问题(测试结果显示，下层智能体执行时没有角速度变化)
"""
import sys
import os
sys.path. insert(0, 'src')

import numpy as np
import torch
from environment. simulator import RobotSimulator
from td3.agent import TD3Agent

def get_state(obs):
    """
    构建状态向量（必须与训练时完全一致）

    状态组成（24维）:
    - 压缩激光数据: 20维
    - 机器人状态: 2维 (distance_to_goal, angle_to_goal)
    - 当前速度: 2维 (current_linear_vel, current_angular_vel)
    """
    laser_data = obs['laser']
    laser_compressed = []

    # 压缩激光数据到 20 维
    points_per_sector = len(laser_data) // 20
    for i in range(20):
        start = i * points_per_sector
        end = (i + 1) * points_per_sector
        sector_min = min(laser_data[start:end])
        laser_compressed.append(sector_min / 10.0)  # 归一化

    # 获取当前速度
    if 'velocity' in obs:
        velocity = obs['velocity']
    else:
        velocity = [0.0, 0.0]

    # 组合状态
    state = np.concatenate([
        laser_compressed,      # 20 维
        obs['robot_state'],    # 2 维 (distance, angle)
        velocity               # 2 维 (当前速度)
    ])

    return state

# 加载模型
print("=" * 60)
print("初始化环境和模型")
print("=" * 60)

env = RobotSimulator()
agent = TD3Agent(state_dim=24, action_dim=2, device=torch.device('cpu'))

model_path = "models/TD3_velodyne_best"
if os.path.exists(f"{model_path}.pth"):
    agent.load(model_path)
    print(f"✓ 模型已加载: {model_path}")
else:
    print(f"✗ 模型不存在: {model_path}")
    exit(1)

# 重置环境
obs = env. reset()

# 构建状态
state = get_state(obs)

print(f"\n状态信息:")
print(f"  激光数据原始长度: {len(obs['laser'])}")
print(f"  激光数据压缩后: 20维")
print(f"  机器人状态: {obs['robot_state']} (2维)")
print(f"  当前速度: {obs. get('velocity', [0.0, 0.0])} (2维)")  # 修改
print(f"  总状态维度: {state.shape} ✓")
print(f"  距离目标: {obs['robot_state'][0]:.2f}m")
print(f"  目标角度: {np.degrees(obs['robot_state'][1]):.1f}°")

# 获取动作
action = agent. get_action(state, add_noise=False)

print("\n" + "=" * 60)
print("动作诊断")
print("=" * 60)
print(f"模型输出动作: {action}")
print(f"  Linear velocity (原始):  {action[0]:.4f}")
print(f"  Angular velocity (原始): {action[1]:.4f}")
print()

# 测试不同的转换方式
print("转换方式 A: [(action[0] + 1) / 2, action[1]]")
action_A = [(action[0] + 1) / 2, action[1]]
print(f"  转换后: [{action_A[0]:.4f}, {action_A[1]:.4f}]")
print(f"  Linear:  {action_A[0]:.4f}  (范围 [0, 1])")
print(f"  Angular: {action_A[1]:.4f}  (范围 [-1, 1])")
print()

print("转换方式 B: [(action[0] + 1) / 2, (action[1] + 1) / 2]")
action_B = [(action[0] + 1) / 2, (action[1] + 1) / 2]
print(f"  转换后: [{action_B[0]:.4f}, {action_B[1]:.4f}]")
print(f"  Linear:  {action_B[0]:.4f}  (范围 [0, 1])")
print(f"  Angular: {action_B[1]:.4f}  (范围 [0, 1])")
print()

print("转换方式 C: 不转换 [action[0], action[1]]")
action_C = [action[0], action[1]]
print(f"  转换后: [{action_C[0]:.4f}, {action_C[1]:.4f}]")
print(f"  Linear:  {action_C[0]:.4f}  (范围 [-1, 1])")
print(f"  Angular: {action_C[1]:.4f}  (范围 [-1, 1])")
print()

# 测试环境响应
print("=" * 60)
print("测试环境响应（执行5步，观察朝向变化）")
print("=" * 60)

initial_x = env.x
initial_y = env.y
initial_theta = env.theta
print(f"初始位置: ({initial_x:.2f}, {initial_y:.2f})")
print(f"初始朝向: {np.degrees(initial_theta):.2f}°")

for method_name, action_in in [("方式A", action_A), ("方式B", action_B), ("方式C", action_C)]:
    # 创建新环境测试
    env_test = RobotSimulator()
    obs_test = env_test.reset()
    # 设置相同的初始状态
    env_test.x = initial_x
    env_test.y = initial_y
    env_test.theta = initial_theta
    env_test.goal_x = env.goal_x
    env_test. goal_y = env.goal_y

    print(f"\n【测试 {method_name}】动作输入: [{action_in[0]:.4f}, {action_in[1]:.4f}]")

    for step in range(5):
        obs_test, reward, done, info = env_test.step(action_in)

        theta_change = np.degrees(env_test.theta - initial_theta)
        x_change = env_test.x - initial_x
        y_change = env_test.y - initial_y

        print(f"  步骤 {step+1}: "
              f"θ变化={theta_change:+7.2f}°, "
              f"位置=({env_test.x:+6.2f}, {env_test.y:+6.2f}), "
              f"移动距离={np.sqrt(x_change**2 + y_change**2):.3f}m")

        if done:
            print(f"    → Episode 结束 (碰撞或到达)")
            break

    # 总结
    total_theta_change = np.degrees(env_test.theta - initial_theta)
    total_distance = np.sqrt((env_test.x - initial_x)**2 + (env_test.y - initial_y)**2)

    if abs(total_theta_change) < 1.0:
        print(f"  ⚠ 警告: 朝向几乎没有变化!  角速度可能未生效!")
    else:
        print(f"  ✓ 朝向变化正常: {total_theta_change:+.2f}°")

    print(f"  总移动距离: {total_distance:.3f}m")

# ========== 新增：验证速度追踪 ==========
print("\n" + "=" * 60)
print("速度追踪验证")
print("=" * 60)

env_verify = RobotSimulator()
obs_verify = env_verify.reset()

print(f"初始观测包含 velocity: {'velocity' in obs_verify}")
if 'velocity' in obs_verify:
    print(f"初始速度: {obs_verify['velocity']}")

    # 执行一个动作
    test_action = [0.5, 0.3]
    obs_next, _, _, _ = env_verify.step(test_action)

    print(f"\n执行动作 {test_action}:")
    print(f"  下一步观测包含 velocity: {'velocity' in obs_next}")
    print(f"  实际速度: {obs_next['velocity']}")
    print(f"  预期速度: [{test_action[0] * 0.5:.3f}, {test_action[1] * 2.0:.3f}]")

    # 验证
    expected = [test_action[0] * 0.5, test_action[1] * 2.0]
    actual = obs_next['velocity']
    if abs(actual[0] - expected[0]) < 0.001 and abs(actual[1] - expected[1]) < 0.001:
        print(f"  ✓ 速度追踪正常")
    else:
        print(f"  ✗ 速度追踪异常")
else:
    print(f"✗ 错误：环境观测缺少 velocity 字段！")
    print(f"   请确保已修改 simulator.py 的 get_observation 方法")
# ==========================================

print("\n" + "=" * 60)
print("诊断结论")
print("=" * 60)