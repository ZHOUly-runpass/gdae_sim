"""
TD3 测试脚本
加载训练好的模型进行测试
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '. .')))

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

from gdae_td3.src.environment.simulator import RobotSimulator
from gdae_td3.src.td3.agent import TD3Agent


class TD3Tester:
    """TD3 测试器"""

    def __init__(self, model_path, render=True):
        """
        初始化测试器

        Args:
            model_path: 模型路径（不含 .pth 扩展名）
            render: 是否可视化
        """
        self.model_path = model_path
        self.render = render

        # 环境配置
        self.map_size = 10.0
        self.laser_range = 5.0
        self.laser_dim = 20
        self.state_dim = 24
        self.action_dim = 2

        # 创建环境
        print("创建测试环境...")
        self.env = RobotSimulator(
            map_size=self.map_size,
            laser_range=self.laser_range,
            laser_dim=self.laser_dim,
            velocity_limits=(0.5, 2.0),
            time_step=0.1
        )

        # 创建智能体
        print("创建 TD3 智能体...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = TD3Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=device
        )

        # 加载模型
        print(f"加载模型: {model_path}")
        self.agent.load(model_path)

        # 可视化设置
        if self.render:
            self.fig, (self.ax_env, self.ax_laser) = plt.subplots(1, 2, figsize=(16, 8))
            self.trajectory = []

        print("初始化完成！\n")

    def _get_state(self, obs, last_action):
        """构建状态向量"""
        laser_data = obs['laser']
        laser_compressed = []

        points_per_sector = len(laser_data) // 20
        for i in range(20):
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

    def test_episode(self, max_steps=500, visualize=True):
        """
        测试单个 episode

        Args:
            max_steps: 最大步数
            visualize: 是否可视化

        Returns:
            dict: 测试结果
        """
        obs = self.env.reset()
        last_action = np.array([0.0, 0.0])
        state = self._get_state(obs, last_action)

        episode_reward = 0
        steps = 0
        done = False

        self.trajectory = [[self.env.x, self.env.y]]

        print(f"初始位置: ({self.env.x:. 2f}, {self.env.y:.2f})")
        print(f"目标位置: ({self.env.goal_x:.2f}, {self.env.goal_y:.2f})")
        print(f"初始距离: {obs['robot_state'][0]:.2f}m\n")

        while not done and steps < max_steps:
            # 选择动作（不添加噪声）
            action = self.agent.get_action(state, add_noise=False)
            action_in = [(action[0] + 1) / 2, action[1]]

            # 执行动作
            next_obs, reward, done, info = self.env.step(action_in)
            next_state = self._get_state(next_obs, action)

            # 记录轨迹
            self.trajectory.append([self.env.x, self.env.y])

            # 可视化
            if visualize and self.render and steps % 5 == 0:
                self._render_frame(next_obs)

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
            'success': info.get('distance_to_goal', 1.0) < 0.3,
            'collision': info.get('collision', False),
            'final_distance': info.get('distance_to_goal', 0.0),
            'trajectory': np.array(self.trajectory)
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
        print("=" * 60 + "\n")

        return result

    def _render_frame(self, obs):
        """渲染单帧"""
        # 清空画布
        self.ax_env.clear()
        self.ax_laser.clear()

        # 绘制环境
        self.ax_env.set_xlim(-self.map_size / 2, self.map_size / 2)
        self.ax_env.set_ylim(-self.map_size / 2, self.map_size / 2)
        self.ax_env.set_aspect('equal')
        self.ax_env.set_title('Environment')
        self.ax_env.grid(True, alpha=0.3)

        # 绘制障碍物
        for obstacle in self.env.obstacles.obstacles:
            circle = Circle(
                (obstacle['x'], obstacle['y']),
                obstacle['radius'],
                color='gray',
                alpha=0.7
            )
            self.ax_env.add_patch(circle)

        # 绘制机器人
        robot = Circle((self.env.x, self.env.y), 0.2, color='blue', alpha=0.8, zorder=10)
        self.ax_env.add_patch(robot)

        # 绘制朝向
        dx = 0.3 * np.cos(self.env.theta)
        dy = 0.3 * np.sin(self.env.theta)
        self.ax_env.arrow(
            self.env.x, self.env.y, dx, dy,
            head_width=0.15, head_length=0.1,
            fc='darkblue', ec='darkblue', zorder=11
        )

        # 绘制目标
        goal = Circle(
            (self.env.goal_x, self.env.goal_y),
            0.15,
            color='green',
            alpha=0.8,
            zorder=10
        )
        self.ax_env.add_patch(goal)

        # 绘制轨迹
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax_env.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.5, linewidth=2)

        # 绘制激光雷达
        laser_data = obs['laser']
        angles = np.linspace(-np.pi / 2, np.pi / 2, len(laser_data))

        self.ax_laser.clear()
        ax_polar = plt.subplot(122, projection='polar')
        ax_polar.plot(angles, laser_data, 'b-', linewidth=1)
        ax_polar.fill(angles, laser_data, 'blue', alpha=0.3)
        ax_polar.set_ylim(0, self.laser_range)
        ax_polar.set_title('Lidar Scan')

        plt.pause(0.01)

    def test_multiple_episodes(self, num_episodes=10):
        """
        测试多个 episodes

        Args:
            num_episodes: 测试数量

        Returns:
            dict: 统计结果
        """
        print("=" * 80)
        print(f"开始测试 {num_episodes} 个 episodes")
        print("=" * 80 + "\n")

        results = []
        success_count = 0
        collision_count = 0
        total_steps = 0
        total_rewards = []

        for i in range(num_episodes):
            print(f"\n--- Episode {i + 1}/{num_episodes} ---")
            result = self.test_episode(visualize=(i == 0))  # 只可视化第一个episode

            results.append(result)
            if result['success']:
                success_count += 1
            if result['collision']:
                collision_count += 1
            total_steps += result['steps']
            total_rewards.append(result['reward'])

        # 统计
        stats = {
            'num_episodes': num_episodes,
            'success_rate': success_count / num_episodes,
            'collision_rate': collision_count / num_episodes,
            'avg_steps': total_steps / num_episodes,
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'results': results
        }

        # 打印统计
        print("\n" + "=" * 80)
        print("测试统计")
        print("=" * 80)
        print(f"总 Episodes: {num_episodes}")
        print(f"成功率: {stats['success_rate'] * 100:.1f}%")
        print(f"碰撞率: {stats['collision_rate'] * 100:.1f}%")
        print(f"平均步数: {stats['avg_steps']:.1f}")
        print(f"平均奖励: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print("=" * 80)

        return stats

    def plot_trajectories(self, results):
        """绘制所有轨迹"""
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlim(-self.map_size / 2, self.map_size / 2)
        ax.set_ylim(-self.map_size / 2, self.map_size / 2)
        ax.set_aspect('equal')
        ax.set_title('All Trajectories')
        ax.grid(True, alpha=0.3)

        # 绘制障碍物
        for obstacle in self.env.obstacles.obstacles:
            circle = Circle(
                (obstacle['x'], obstacle['y']),
                obstacle['radius'],
                color='gray',
                alpha=0.5
            )
            ax.add_patch(circle)

        # 绘制所有轨迹
        for i, result in enumerate(results):
            traj = result['trajectory']
            color = 'green' if result['success'] else 'red'
            alpha = 0.3
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=1)

        plt.savefig('trajectories.png', dpi=150, bbox_inches='tight')
        print("轨迹图保存至: trajectories.png")
        plt.show()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Test TD3 model')
    parser.add_argument('--model', type=str, default='models/TD3_velodyne_best',
                        help='Path to model (without .pth extension)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable visualization')

    args = parser.parse_args()

    # 创建测试器
    tester = TD3Tester(args.model, render=not args.no_render)

    # 测试
    stats = tester.test_multiple_episodes(num_episodes=args.episodes)

    # 绘制轨迹
    if not args.no_render:
        tester.plot_trajectories(stats['results'])


if __name__ == "__main__":
    main()