"""
诊断智能体原地打转问题
分析动作输出、状态构建、环境交互的完整流程
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from environment.simulator import RobotSimulator
from td3.agent import TD3Agent


def get_state(obs, laser_dim=20):
    """
    构建状态向量（与训练时一致）
    状态组成: [laser(20) + distance(1) + theta(1) + action(2)] = 24维
    """
    laser_data = obs['laser']
    laser_compressed = []
    
    # 压缩激光数据
    points_per_sector = len(laser_data) // laser_dim
    for i in range(laser_dim):
        start = i * points_per_sector
        end = (i + 1) * points_per_sector
        sector_min = min(laser_data[start:end])
        # 注意：这里除以10.0是归一化，但训练代码中说"直接使用，不归一化"
        laser_compressed.append(sector_min / 10.0)
    
    # 机器人状态
    robot_state = obs['robot_state']
    
    # 当前动作
    action = obs['action']
    
    state = np.concatenate([
        laser_compressed,
        robot_state,
        action
    ])
    
    return state


def diagnose_model_output(agent, env, num_steps=10):
    """诊断模型输出的动作"""
    print("\n" + "="*80)
    print("📊 第一部分：模型动作输出诊断")
    print("="*80)
    
    obs = env.reset()
    state = get_state(obs)
    
    actions_linear = []
    actions_angular = []
    
    print(f"\n初始状态信息:")
    print(f"  机器人位置: ({env.x:.2f}, {env.y:.2f})")
    print(f"  目标位置: ({env.goal_x:.2f}, {env.goal_y:.2f})")
    print(f"  距离目标: {obs['robot_state'][0]:.2f}m")
    print(f"  目标角度: {np.degrees(obs['robot_state'][1]):.1f}°")
    print(f"  状态维度: {state.shape}")
    
    print(f"\n连续{num_steps}步的动作输出:")
    print("-" * 80)
    print(f"{'步数':<6} {'线速度':<12} {'角速度':<12} {'转换后线速度':<15} {'转换后角速度':<15}")
    print("-" * 80)
    
    for step in range(num_steps):
        # 获取原始动作
        action_raw = agent.get_action(state, add_noise=False)
        
        # 转换动作（与训练代码一致）
        action_converted = [(action_raw[0] + 1) / 2, action_raw[1]]
        
        actions_linear.append(action_raw[0])
        actions_angular.append(action_raw[1])
        
        print(f"{step+1:<6} {action_raw[0]:>+.6f}  {action_raw[1]:>+.6f}  "
              f"{action_converted[0]:>.6f}       {action_converted[1]:>+.6f}")
        
        # 执行动作
        next_obs, reward, done, info = env.step(action_converted)
        next_state = get_state(next_obs)
        
        state = next_state
        obs = next_obs
    
    print("-" * 80)
    
    # 统计分析
    print(f"\n📈 动作统计分析:")
    print(f"  线速度 (原始):  均值={np.mean(actions_linear):+.4f}, "
          f"标准差={np.std(actions_linear):.4f}, "
          f"范围=[{np.min(actions_linear):+.4f}, {np.max(actions_linear):+.4f}]")
    print(f"  角速度 (原始):  均值={np.mean(actions_angular):+.4f}, "
          f"标准差={np.std(actions_angular):.4f}, "
          f"范围=[{np.min(actions_angular):+.4f}, {np.max(actions_angular):+.4f}]")
    
    # 检查是否卡在极值
    if np.abs(np.mean(actions_angular)) > 0.9:
        print(f"\n⚠️  警告: 角速度平均值接近极限值 ({np.mean(actions_angular):+.4f})")
        print(f"    这表明模型可能输出了饱和的角速度，导致持续旋转！")
    
    if np.std(actions_angular) < 0.1:
        print(f"\n⚠️  警告: 角速度标准差很小 ({np.std(actions_angular):.4f})")
        print(f"    这表明模型输出的角速度几乎不变化，可能陷入了固定策略！")
    
    return actions_linear, actions_angular


def diagnose_state_construction(env):
    """诊断状态构建的一致性"""
    print("\n" + "="*80)
    print("🔍 第二部分：状态构建一致性诊断")
    print("="*80)
    
    obs = env.reset()
    
    print(f"\n环境返回的观测值:")
    print(f"  laser: 长度={len(obs['laser'])}, 范围=[{min(obs['laser']):.2f}, {max(obs['laser']):.2f}]")
    print(f"  robot_state: {obs['robot_state']}")
    print(f"  action: {obs['action']}")
    
    # 方式1: 压缩激光 + 归一化 (除以10)
    state_v1 = get_state(obs, laser_dim=20)
    
    # 方式2: 压缩激光 + 不归一化
    laser_data = obs['laser']
    laser_compressed_v2 = []
    points_per_sector = len(laser_data) // 20
    for i in range(20):
        start = i * points_per_sector
        end = (i + 1) * points_per_sector
        sector_min = min(laser_data[start:end])
        laser_compressed_v2.append(sector_min)  # 不除以10
    
    state_v2 = np.concatenate([
        laser_compressed_v2,
        obs['robot_state'],
        obs['action']
    ])
    
    print(f"\n状态构建方式对比:")
    print(f"  方式1 (激光/10): 激光范围=[{min(state_v1[:20]):.2f}, {max(state_v1[:20]):.2f}]")
    print(f"  方式2 (激光原值): 激光范围=[{min(state_v2[:20]):.2f}, {max(state_v2[:20]):.2f}]")
    
    # 检查训练代码的说明
    print(f"\n📝 训练代码中的注释:")
    print(f"  train_td3.py 第121行: '激光数据：直接使用，不归一化'")
    print(f"  但 quick_visualize.py 中使用: sector_min / 10.0")
    
    print(f"\n⚠️  潜在问题:")
    if abs(min(state_v1[:20]) - min(state_v2[:20])) > 0.01:
        print(f"  ✗ 状态构建不一致！训练时可能用的是方���2，但测试用的是方式1")
        print(f"  ✗ 这会导致模型接收到完全不同范围的输入，策略失效！")
    else:
        print(f"  ✓ 状态构建一致")
    
    return state_v1, state_v2


def diagnose_action_conversion(env):
    """诊断动作转换逻辑"""
    print("\n" + "="*80)
    print("🔄 第三部分：动作转换逻辑诊断")
    print("="*80)
    
    print(f"\n环境速度限制:")
    print(f"  max_linear_vel: {env.max_linear_vel} m/s")
    print(f"  max_angular_vel: {env.max_angular_vel} rad/s")
    
    # 测试不同的动作转换方式
    test_actions = [
        np.array([1.0, 1.0]),    # 最大正值
        np.array([-1.0, -1.0]),  # 最大负值
        np.array([0.0, 0.0]),    # 零值
        np.array([0.5, -0.5]),   # 混合值
    ]
    
    print(f"\n动作转换测试:")
    print("-" * 80)
    print(f"{'模型输出':<25} {'当前转换':<30} {'实际速度':<30}")
    print("-" * 80)
    
    for action in test_actions:
        # 当前使用的转换方式
        action_converted = [(action[0] + 1) / 2, action[1]]
        
        # 计算实际速度
        actual_linear = action_converted[0] * env.max_linear_vel
        actual_angular = action_converted[1] * env.max_angular_vel
        
        print(f"[{action[0]:+.2f}, {action[1]:+.2f}]          "
              f"[{action_converted[0]:.2f}, {action_converted[1]:+.2f}]                "
              f"linear={actual_linear:.2f}, angular={actual_angular:+.2f}")
    
    print("-" * 80)
    
    print(f"\n⚠️  问题分析:")
    print(f"  当前转换: action_in = [(action[0] + 1) / 2, action[1]]")
    print(f"  - 线速度: [-1, 1] → [0, 1] → [0, {env.max_linear_vel}] m/s  ✓ 正确")
    print(f"  - 角速度: [-1, 1] → [-1, 1] → [{-env.max_angular_vel}, {env.max_angular_vel}] rad/s")
    
    print(f"\n  如果模型始终输出角速度≈1.0:")
    print(f"  → 实际角速度 = 1.0 * {env.max_angular_vel} = {env.max_angular_vel} rad/s")
    print(f"  → 每秒旋转 {np.degrees(env.max_angular_vel):.1f}°")
    print(f"  → 在time_step={env.time_step}s内旋转 {np.degrees(env.max_angular_vel * env.time_step):.1f}°")
    print(f"  → 这会导致机器人快速旋转！")


def diagnose_network_weights(agent):
    """诊断网络权重"""
    print("\n" + "="*80)
    print("🧠 第四部分：神经网络权重诊断")
    print("="*80)
    
    print(f"\nActor 网络结构:")
    for name, param in agent.actor.named_parameters():
        print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, "
              f"std={param.std().item():.4f}")
    
    # 检查输出层偏置
    output_layer = None
    for name, param in agent.actor.named_parameters():
        if 'fc3.bias' in name or 'output' in name or 'action' in name:
            output_layer = param
            print(f"\n⚠️  输出层偏置: {param.detach().cpu().numpy()}")
            
            if torch.abs(param[1]) > 0.5:  # 角速度的偏置
                print(f"  ✗ 角速度输出偏置很大 ({param[1].item():.4f})！")
                print(f"  ✗ 这可能导致模型默认输出大角速度！")


def visualize_trajectory(env, agent, max_steps=100):
    """可视化轨迹"""
    print("\n" + "="*80)
    print("📍 第五部分：轨迹可视化")
    print("="*80)
    
    obs = env.reset()
    state = get_state(obs)
    
    positions = [(env.x, env.y)]
    angles = [env.theta]
    distances = [obs['robot_state'][0]]
    actions_linear = []
    actions_angular = []
    
    for step in range(max_steps):
        action = agent.get_action(state, add_noise=False)
        action_in = [(action[0] + 1) / 2, action[1]]
        
        actions_linear.append(action[0])
        actions_angular.append(action[1])
        
        next_obs, reward, done, info = env.step(action_in)
        next_state = get_state(next_obs)
        
        positions.append((env.x, env.y))
        angles.append(env.theta)
        distances.append(next_obs['robot_state'][0])
        
        state = next_state
        
        if done:
            break
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 轨迹图
    ax1 = axes[0, 0]
    positions = np.array(positions)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(env.goal_x, env.goal_y, 'r*', markersize=15, label='Goal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Robot Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # 2. 距离变化
    ax2 = axes[0, 1]
    ax2.plot(distances, 'b-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance to Goal (m)')
    ax2.set_title('Distance over Time')
    ax2.grid(True)
    
    # 3. 线速度
    ax3 = axes[1, 0]
    ax3.plot(actions_linear, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Linear Velocity (normalized)')
    ax3.set_title('Linear Velocity Commands')
    ax3.set_ylim([-1.1, 1.1])
    ax3.grid(True)
    
    # 4. 角速度
    ax4 = axes[1, 1]
    ax4.plot(actions_angular, 'r-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Angular Velocity (normalized)')
    ax4.set_title('Angular Velocity Commands')
    ax4.set_ylim([-1.1, 1.1])
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('diagnosis_trajectory.png', dpi=150)
    print(f"\n✓ 轨迹图已保存: diagnosis_trajectory.png")
    
    # 分析轨迹
    total_distance_moved = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    initial_distance = distances[0]
    final_distance = distances[-1]
    
    print(f"\n轨迹统计:")
    print(f"  总步数: {len(positions)-1}")
    print(f"  初始距离: {initial_distance:.2f}m")
    print(f"  最终距离: {final_distance:.2f}m")
    print(f"  距离变化: {initial_distance - final_distance:+.2f}m")
    print(f"  实际移动距离: {total_distance_moved:.2f}m")
    
    if final_distance > initial_distance * 0.9:
        print(f"\n⚠️  警告: 几乎没有接近目标！")
    
    if total_distance_moved > initial_distance * 3:
        print(f"\n⚠️  警告: 移动距离远大于直线距离，可能在打转！")


def main():
    """主诊断函数"""
    print("="*80)
    print("🔧 TD3 智能体原地打转问题诊断工具")
    print("="*80)
    
    model_path = "gdae_td3/src/training/models/TD3_velodyne_best"
    
    # 检查模型文件
    if not os.path.exists(f"{model_path}.pth"):
        print(f"\n✗ 模型文件不存在: {model_path}.pth")
        print("请检查模型路径是否正确")
        return
    
    print(f"\n✓ 找到模型文件: {model_path}.pth")
    
    # 创建环境
    print("\n创建环境...")
    env = RobotSimulator(
        map_size=10.0,
        laser_range=5.0,
        laser_dim=20,
        velocity_limits=(0.5, 2.0),
        time_step=0.1
    )
    
    # 创建智能体
    print("创建智能体...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3Agent(
        state_dim=24,
        action_dim=2,
        max_action=1.0,
        device=device
    )
    
    # 加载模型
    print("加载模型...")
    agent.load(model_path)
    print("✓ 模型加载成功")
    
    # 执行诊断
    try:
        # 1. 模型输出诊断
        actions_linear, actions_angular = diagnose_model_output(agent, env, num_steps=20)
        
        # 2. 状态构建诊断
        state_v1, state_v2 = diagnose_state_construction(env)
        
        # 3. 动作转换诊断
        diagnose_action_conversion(env)
        
        # 4. 网络权重诊断
        diagnose_network_weights(agent)
        
        # 5. 轨迹可视化
        visualize_trajectory(env, agent, max_steps=200)
        
        # 总结
        print("\n" + "="*80)
        print("📋 诊断总结与建议")
        print("="*80)
        
        print("\n🔍 已检查的问题点:")
        print("  1. ✓ 模型动作输出范围")
        print("  2. ✓ 状态构建一致性")
        print("  3. ✓ 动作转换逻辑")
        print("  4. ✓ 网络权重分布")
        print("  5. ✓ 实际运行轨迹")
        
        print("\n💡 可能的解决方案:")
        print("  1. 检查 quick_visualize.py 中的状态构建是否与 train_td3.py 一致")
        print("  2. 确认激光数据是否需要归一化 (除以10)")
        print("  3. 验证动作转换公式是否正确")
        print("  4. 尝试重新训练模型，确保训练和测试环境完全一致")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
