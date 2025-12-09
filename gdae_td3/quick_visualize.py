"""
快速可视化脚本
一键启动可视化，无需命令行参数
"""
import sys
import os
sys.path.insert(0, os. path.abspath(os.path.join(os. path.dirname(__file__), 'src')))

import numpy as np
import torch

from environment. simulator import RobotSimulator
from td3.agent import TD3Agent
from utils.visualizer import TD3Visualizer


def get_state(obs, last_action=None):
    """
    构建状态向量，与训练阶段完全一致。
    """
    laser_data = obs['laser']
    laser_compressed = []

    points_per_sector = len(laser_data) // 20
    for i in range(20):
        start = i * points_per_sector
        end = (i + 1) * points_per_sector
        sector_min = min(laser_data[start: end])
        laser_compressed.append(sector_min / 10.0)

    # 修改：使用 last_action
    if last_action is None:
        last_action = np.array([0.0, 0.0])

    state = np.concatenate([
        laser_compressed,
        obs['robot_state'],
        last_action  # 使用上一步动作
    ])

    return state


def main():
    """主函数"""
    print("=" * 80)
    print("TD3 快速可视化")
    print("=" * 80)

    # ===== 修复：正确的模型路径 =====
    # 获取脚本所在目录的绝对路径
    script_dir = os.path. dirname(os.path.abspath(__file__))

    # 模型在 src/training/models/ 目录下
    MODEL_PATH = os. path.join(script_dir, "src", "training", "models", "TD3_velodyne_best")

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
        velocity_limits=(0.5, 2.0),
        time_step=0.1
    )

    print("创建智能体...")
    device = torch.device("cuda" if torch. cuda.is_available() else "cpu")
    agent = TD3Agent(state_dim=24, action_dim=2, device=device)

    # 加载模型（如果存在）
    model_file = f"{MODEL_PATH}.pth"
    if os.path. exists(model_file):
        print(f"✓ 找到模型文件: {model_file}")
        print(f"加载模型: {MODEL_PATH}")
        agent.load(MODEL_PATH)
        print("✓ 模型加载成功")
    else:
        print(f"⚠ 模型不存在: {model_file}")
        print("使用随机策略")
        # 列出可能的模型文件位置
        possible_dirs = [
            os.path.join(script_dir, "models"),
            os.path.join(script_dir, "src", "training", "models"),
            "models",
        ]
        print("\n检查可能的模型目录:")
        for d in possible_dirs:
            if os. path.exists(d):
                files = os.listdir(d)
                pth_files = [f for f in files if f.endswith('.pth')]
                print(f"  {d}: {pth_files if pth_files else '(无 . pth 文件)'}")
            else:
                print(f"  {d}: (目录不存在)")

    # 创建可视化器
    print("创建可视化器...")
    visualizer = TD3Visualizer(env, agent)

    # 开始运行 episodes
    for ep in range(NUM_EPISODES):
        # 重置环境
        obs = env.reset()

        # 只在第一个 Episode 后调用 visualizer. reset()
        if ep > 0:
            visualizer.reset()

        last_action = np. array([0.0, 0.0])
        state = get_state(obs, last_action=last_action)
        done = False
        steps = 0
        episode_reward = 0
        trajectory = []
        trajectory. append((env.x, env.y))

        print("=" * 60)
        print(f"Episode {visualizer.episode_count + 1}")
        print(f"初始位置: ({env.x:.2f}, {env. y:.2f})")
        print(f"目标位置: ({env.goal_x:.2f}, {env.goal_y:.2f})")
        print(f"初始距离: {obs['robot_state'][0]:.2f}m")
        print("=" * 60)

        while not done and steps < MAX_STEPS:
            action = agent.get_action(state, add_noise=False)
            action_in = [(action[0] + 1) / 2, action[1]]

            next_obs, reward, done, info = env.step(action_in)
            trajectory.append((env. x, env.y))

            next_state = get_state(next_obs, last_action=action)

            visualizer.update(obs, action_in, reward)

            obs = next_obs  # 重要：更新 obs
            state = next_state
            last_action = action
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

        # 打印本轮轨迹
        print("\n本轮运行轨迹 (Trajectory):")
        formatted_traj = [f"({x:.2f}, {y:.2f})" for x, y in trajectory]
        print(", ".join(formatted_traj))
        print("-" * 60 + "\n")

        # 等待下一 Episode
        if ep < NUM_EPISODES - 1:
            input("按 Enter 继续下一个 Episode...")

    print("\n可视化完成！关闭窗口退出...")
    visualizer.show()


if __name__ == "__main__":
    main()