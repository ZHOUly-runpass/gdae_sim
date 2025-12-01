"""
TD3 可视化启动脚本
实时展示训练好的 TD3 模型的导航和避障效果
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import torch
import argparse

from gdae_td3.src.environment.simulator import RobotSimulator
from gdae_td3.src.td3.agent import TD3Agent
from gdae_td3.src.utils.visualizer import TD3Visualizer
from gdae_td3.src.utils. video_recorder import VideoRecorder


def get_state(obs, last_action, laser_dim=20):
    """构建状态向量"""
    laser_data = obs['laser']
    laser_compressed = []

    points_per_sector = len(laser_data) // laser_dim
    for i in range(laser_dim):
        start = i * points_per_sector
        end = (i + 1) * points_per_sector
        sector_min = min(laser_data[start:end])
        laser_compressed.append(sector_min / 10.0)

    state = np.concatenate([
        laser_compressed,
        obs['robot_state'],
        last_action
    ])

    return state


def run_episode(env, agent, visualizer, max_steps=500, record_video=False):
    """
    运行单个 episode

    Args:
        env: 环境
        agent: TD3 智能体
        visualizer: 可视化器
        max_steps: 最大步数
        record_video: 是否录制视频

    Returns:
        dict: Episode 结果
    """
    # 重置环境和可视化器
    obs = env.reset()
    visualizer.reset()

    if record_video:
        recorder = VideoRecorder(visualizer, fps=20)

    last_action = np.array([0.0, 0.0])
    state = get_state(obs, last_action)

    episode_reward = 0
    steps = 0
    done = False

    print(f"\nEpisode {visualizer.episode_count}")
    print(f"初始位置: ({env.x:. 2f}, {env.y:. 2f})")
    print(f"目标位置: ({env.goal_x:.2f}, {env.goal_y:.2f})")
    print(f"初始距离: {obs['robot_state'][0]:. 2f}m\n")

    while not done and steps < max_steps:
        # 使用 TD3 选择动作
        action = agent.get_action(state, add_noise=False)
        action_in = [(action[0] + 1) / 2, action[1]]

        # 执行动作
        next_obs, reward, done, info = env.step(action_in)
        next_state = get_state(next_obs, action)

        # 更新可视化
        visualizer.update(obs, action, reward)

        # 录制视频帧
        if record_video:
            recorder.capture_frame()

        # 更新状态
        state = next_state
        last_action = action
        episode_reward += reward
        steps += 1

        # 打印进度
        if steps % 50 == 0:
            print(f"步数: {steps}, 累计奖励: {episode_reward:.2f}, "
                  f"距离目标: {info['distance_to_goal']:.2f}m")

    # 结果统计
    result = {
        'steps': steps,
        'reward': episode_reward,
        'success': info. get('distance_to_goal', 1.0) < 0.3,
        'collision': info.get('collision', False),
        'final_distance': info.get('distance_to_goal', 0.0),
    }

    print("\n" + "=" * 60)
    if result['success']:
        print("✓ 成功到达目标！")
    elif result['collision']:
        print("✗ 发生碰撞")
    else:
        print("⏱ 超时")
    print(f"步数: {steps}")
    print(f"总奖励: {episode_reward:.2f}")
    print(f"最终距离: {info['distance_to_goal']:. 2f}m")
    print("=" * 60)

    # 保存视频
    if record_video:
        video_path = f"episode_{visualizer.episode_count}. mp4"
        recorder.save_video(video_path)

        gif_path = f"episode_{visualizer.episode_count}.gif"
        recorder.save_gif(gif_path)

    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Visualize TD3 Navigation')
    parser.add_argument('--model', type=str, default='models/TD3_velodyne_best',
                       help='Path to model (without . pth extension)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to visualize')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--record-video', action='store_true',
                       help='Record video for each episode')
    parser.add_argument('--save-fig', action='store_true',
                       help='Save final figure')

    args = parser.parse_args()

    print("=" * 80)
    print("TD3 Navigation Visualization")
    print("=" * 80)
    print(f"模型路径: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"最大步数: {args.max_steps}")
    print(f"录制视频: {args.record_video}")
    print("=" * 80)

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
    print("创建 TD3 智能体...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3Agent(state_dim=24, action_dim=2, device=device)

    # 加载模型
    print(f"加载模型: {args.model}")
    try:
        agent.load(args.model)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        print("使用未训练的模型（随机策略）")

    # 创建可视化器
    print("创建可视化器...")
    visualizer = TD3Visualizer(env, agent, figsize=(18, 10))

    # 运行多个 episodes
    results = []
    for ep in range(args.episodes):
        result = run_episode(
            env, agent, visualizer,
            max_steps=args. max_steps,
            record_video=args.record_video
        )
        results.append(result)

        # 等待用户确认继续
        if ep < args.episodes - 1:
            input("\n按 Enter 继续下一个 episode...")

    # 保存最终图形
    if args.save_fig:
        visualizer.save_figure('td3_final_visualization.png')

    # 统计结果
    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80)
    success_count = sum(1 for r in results if r['success'])
    collision_count = sum(1 for r in results if r['collision'])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])

    print(f"总 Episodes: {args.episodes}")
    print(f"成功率: {success_count/args.episodes*100:.1f}%")
    print(f"碰撞率: {collision_count/args.episodes*100:. 1f}%")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"平均奖励: {avg_reward:.2f}")
    print("=" * 80)

    # 显示图形
    visualizer.show()


if __name__ == "__main__":
    main()