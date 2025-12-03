"""
快速可视化脚本
一键启动可视化，无需命令行参数
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import torch

from environment.simulator import RobotSimulator
from td3.agent import TD3Agent
from utils.visualizer import TD3Visualizer


def get_state(obs, last_action):
    """
    构建状态向量，与训练阶段完全一致。

    状态组成（24维）:
    - 压缩激光数据: 20维
    - 机器人状态: 2维 (distance_to_goal, angle_to_goal)
    - 上一步动作: 2维 (last_linear_vel, last_angular_vel)
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

    # 拼接状态：[laser(20), distance(1), angle(1), last_linear(1), last_angular(1)]
    state = np.concatenate([
        laser_compressed,      # 激光数据（20维）
        obs['robot_state'],    # [距离到目标, 相对角度]
        last_action            # [线速度, 角速度]（必须与实际执行的一致）
    ])

    return state


def main():
    """主函数"""
    print("=" * 80)
    print("TD3 快速可视化")
    print("=" * 80)

    # ===== 配置 =====
    MODEL_PATH = "models/TD3_velodyne_best"  # 模型路径
    NUM_EPISODES = 3                        # 测试的 Episode 数量
    MAX_STEPS = 500                         # 每个 Episode 的最大步数
    # =================

    print(f"模型路径: {MODEL_PATH}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"最大步数: {MAX_STEPS}")
    print("=" * 80)

    # 创建环境和智能体
    print("\n创建环境...")
    env = RobotSimulator(
        map_size=10.0,
        laser_range=5.0,
        laser_dim=20,
        velocity_limits=(0.5, 2.0),        # 修复后的环境配置
        time_step=0.1
    )

    print("创建智能体...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3Agent(state_dim=24, action_dim=2, device=device)

    # 加载模型（如果存在）
    if os.path.exists(f"{MODEL_PATH}.pth"):
        print(f"加载模型: {MODEL_PATH}")
        agent.load(MODEL_PATH)
        print("✓ 模型加载成功")
    else:
        print(f"⚠ 模型不存在: {MODEL_PATH}.pth")
        print("使用随机策略")

    # 创建可视化器
    print("创建可视化器...")
    visualizer = TD3Visualizer(env, agent)

    # 开始运行 episodes
    for ep in range(NUM_EPISODES):
        # 重置环境
        obs = env.reset()

        # 只在第一个 Episode 后调用 visualizer.reset()
        if ep > 0:
            visualizer.reset()

        # 初始化状态
        last_action = np.array([0.0, 0.0])  # 必须与 action_in 的范围一致！
        state = get_state(obs, last_action)

        # Episode 设置
        done = False
        steps = 0
        episode_reward = 0

        # --- 新增: 初始化轨迹列表 ---
        trajectory = []
        # 记录初始位置
        trajectory.append((env.x, env.y))
        # --------------------------

        print("=" * 60)
        print(f"Episode {visualizer.episode_count + 1}")
        print(f"初始位置: ({env.x:.2f}, {env.y:.2f})")
        print(f"目标位置: ({env.goal_x:.2f}, {env.goal_y:.2f})")
        print(f"初始距离: {obs['robot_state'][0]:.2f}m")
        print("=" * 60)

        while not done and steps < MAX_STEPS:
            # 使用 TD3 选择动作（无噪声）
            action = agent.get_action(state, add_noise=False)

            # 转换动作到环境实际范围，用于执行
            action_in = [(action[0] + 1) / 2, action[1]]  # 转换线速度到 [0,1]

            # 执行动作
            next_obs, reward, done, info = env.step(action_in)

            # --- 新增: 记录当前步骤后的位置 ---
            trajectory.append((env.x, env.y))
            # -------------------------------

            # 构建新的状态向量
            next_state = get_state(next_obs, np.array(action_in))  # last_action = action_in

            # 更新可视化视图
            visualizer.update(obs, action_in, reward)

            # 更新 episode 状态
            state = next_state  # 更新当前状态
            last_action = np.array(action_in)  # 更新实际执行的动作为历史动作
            episode_reward += reward
            steps += 1

            # 打印进度
            if steps % 50 == 0:
                print(f"步数: {steps}, 奖励: {episode_reward:.2f}, "
                      f"距离: {info['distance_to_goal']:.2f}m")

        # Episode 结束
        print("\n" + "-" * 60)
        if info.get('reach_goal', False):
            print("✓ 成功到达目标！")
        elif info.get('collision', False):
            print("✗ 发生碰撞")
        else:
            print("⏱ 超时")
        print(f"总步数: {steps}")
        print(f"总奖励: {episode_reward:.2f}")

        # --- 新增: 打印本轮轨迹 ---
        print("\n本轮运行轨迹 (Trajectory):")
        # 将轨迹格式化为保留2位小数的字符串列表，方便阅读
        formatted_traj = [f"({x:.2f}, {y:.2f})" for x, y in trajectory]
        print(", ".join(formatted_traj))
        # ------------------------

        print("-" * 60 + "\n")

        # 等待下一 Episode
        if ep < NUM_EPISODES - 1:
            input("按 Enter 继续下一个 Episode...")

    print("\n可视化完成！关闭窗口退出...")
    visualizer.show()


if __name__ == "__main__":
    main()