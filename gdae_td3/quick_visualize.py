"""
快速可视化脚本
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os. path.join(os.path. dirname(__file__), 'src')))

import numpy as np
import torch

from environment.simulator import RobotSimulator
from td3.agent import TD3Agent
from utils.visualizer import TD3Visualizer


def get_state(obs):
    """
    构建状态向量，与训练阶段完全一致
    """
    laser_data = np.array(obs['laser'])
    robot_state = np.array(obs['robot_state'])
    action = np.array(obs['action'])

    state = np.concatenate([laser_data, robot_state, action])
    return state


def main():
    """主函数"""
    print("=" * 80)
    print("TD3 快速可视化")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, "src", "training", "models", "TD3_velodyne_best")

    NUM_EPISODES = 3
    MAX_STEPS = 500

    print(f"模型路径: {MODEL_PATH}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"最大步数: {MAX_STEPS}")
    print("=" * 80)

    print("\n创建环境...")
    env = RobotSimulator(
        map_size=10.0,
        laser_range=5.0,
        laser_dim=20,
        velocity_limits=(0.5, 2.0),
        time_step=0.1
    )

    print("创建智能体...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3Agent(state_dim=24, action_dim=2, device=device)

    model_file = f"{MODEL_PATH}.pth"
    if os.path.exists(model_file):
        print(f"✓ 找到模型文件: {model_file}")
        agent.load(MODEL_PATH)
        print("✓ 模型加载成功")
    else:
        print(f"⚠ 模型不存在: {model_file}")
        print("使用随机策略")

    print("创建可视化器...")
    visualizer = TD3Visualizer(env, agent)

    for ep in range(NUM_EPISODES):
        obs = env.reset()

        if ep > 0:
            visualizer.reset()

        state = get_state(obs)
        done = False
        steps = 0
        episode_reward = 0
        trajectory = [(env.x, env.y)]

        print("=" * 60)
        print(f"Episode {ep + 1}")
        print(f"初始位置: ({env.x:.2f}, {env.y:.2f})")
        print(f"目标位置: ({env.goal_x:.2f}, {env.goal_y:.2f})")
        print(f"初始距离: {obs['robot_state'][0]:.2f}m")
        print("=" * 60)

        while not done and steps < MAX_STEPS:
            action = agent.get_action(state, add_noise=False)
            action_in = [(action[0] + 1) / 2, action[1]]

            next_obs, reward, done, info = env.step(action_in)
            trajectory.append((env.x, env.y))

            next_state = get_state(next_obs)

            visualizer.update(obs, action_in, reward)

            obs = next_obs
            state = next_state
            episode_reward += reward
            steps += 1

            if steps % 50 == 0:
                print(f"步数: {steps}, 奖励: {episode_reward:.2f}, "
                      f"距离:  {info['distance_to_goal']:.2f}m")

        print("\n" + "-" * 60)
        if info. get('reach_goal', False):
            print("✓ 成功到达目标！")
        elif info.get('collision', False):
            print("✗ 发生碰撞")
        else:
            print("⏱ 超时")
        print(f"总步数: {steps}")
        print(f"总奖励: {episode_reward:.2f}")
        print("-" * 60 + "\n")

        if ep < NUM_EPISODES - 1:
            input("按 Enter 继续下一个 Episode...")

    print("\n可视化完成！")
    visualizer.show()


if __name__ == "__main__":
    main()