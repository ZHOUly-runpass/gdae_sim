"""
TD3 算法实时可视化器
支持实时显示机器人导航、避障行为、激光雷达数据等
"""
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端（交互式窗口）

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from collections import deque


class TD3Visualizer:
    """
    TD3 实时可视化器
    显示：环境、机器人、障碍物、激光雷达、轨迹、状态信息
    """

    def __init__(self, env, agent, figsize=(18, 10), update_interval=50):
        """
        初始化可视化器

        Args:
            env: 仿真环境
            agent: TD3 智能体
            figsize: 图形尺寸
            update_interval: 更新间隔（毫秒）
        """
        self. env = env
        self.agent = agent
        self.update_interval = update_interval

        # 创建图形
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle('TD3 Robot Navigation - Real-time Visualization',
                         fontsize=16, fontweight='bold')

        # 创建子图布局
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 主环境视图（左侧大图）
        self.ax_env = self.fig.add_subplot(gs[:, :2])

        # 激光雷达视图（右上）
        self.ax_laser = self.fig.add_subplot(gs[0, 2], projection='polar')

        # Q值和动作视图（右中）
        self.ax_action = self.fig.add_subplot(gs[1, 2])

        # 奖励曲线视图（右下）
        self.ax_reward = self.fig.add_subplot(gs[2, 2])

        # 数据记录
        self.trajectory = []
        self.reward_history = deque(maxlen=200)
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0

        # 历史轨迹（用于显示多个episode）
        self.all_trajectories = []

        # 动作历史
        self.action_history = deque(maxlen=50)

        # 初始化图形元素
        self._init_plots()

    def _init_plots(self):
        """初始化所有子图"""
        # 环境视图设置
        self.ax_env.set_xlim(-self.env.map_size/2 - 1, self.env.map_size/2 + 1)
        self.ax_env.set_ylim(-self. env.map_size/2 - 1, self.env. map_size/2 + 1)
        self.ax_env.set_aspect('equal')
        self.ax_env. set_xlabel('X (m)', fontsize=10)
        self.ax_env.set_ylabel('Y (m)', fontsize=10)
        self.ax_env.set_title('Navigation Environment', fontsize=12, fontweight='bold')
        self.ax_env.grid(True, alpha=0.3, linestyle='--')

        # 激光雷达视图设置
        self. ax_laser.set_ylim(0, self.env. laser_range)
        self. ax_laser.set_title('Lidar Scan', fontsize=10, fontweight='bold', pad=20)
        self. ax_laser.grid(True, alpha=0.3)

        # 动作视图设置
        self.ax_action.set_ylim(-1.2, 1.2)
        self.ax_action.set_xlim(0, 50)
        self.ax_action.set_xlabel('Time Steps', fontsize=9)
        self.ax_action.set_ylabel('Action Value', fontsize=9)
        self.ax_action.set_title('Action History', fontsize=10, fontweight='bold')
        self.ax_action.grid(True, alpha=0.3)
        self.ax_action.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 奖励曲线设置
        self.ax_reward.set_xlim(0, 200)
        self.ax_reward.set_xlabel('Steps', fontsize=9)
        self.ax_reward.set_ylabel('Reward', fontsize=9)
        self.ax_reward.set_title('Reward History', fontsize=10, fontweight='bold')
        self. ax_reward.grid(True, alpha=0.3)
        self.ax_reward.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    def _draw_environment(self, obs, action, reward):
        """绘制环境"""
        self.ax_env.clear()

        # 重新设置范围
        self.ax_env.set_xlim(-self. env.map_size/2 - 1, self.env. map_size/2 + 1)
        self.ax_env.set_ylim(-self.env.map_size/2 - 1, self.env.map_size/2 + 1)
        self.ax_env.set_aspect('equal')
        self.ax_env.grid(True, alpha=0.3, linestyle='--')

        # 绘制地图边界
        border = Rectangle(
            (-self.env.map_size/2, -self.env.map_size/2),
            self.env.map_size, self.env.map_size,
            fill=False, edgecolor='black', linewidth=2
        )
        self.ax_env.add_patch(border)

        # 绘制障碍物
        for obs_obj in self.env.obstacles. obstacles:
            obstacle = Circle(
                (obs_obj['x'], obs_obj['y']),
                obs_obj['radius'],
                facecolor='dimgray',
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5,
                zorder=5
            )
            self.ax_env.add_patch(obstacle)

        # 绘制历史轨迹（浅色）
        for traj in self.all_trajectories[-5:]:
            if len(traj) > 1:
                traj_array = np.array(traj)
                self.ax_env. plot(
                    traj_array[:, 0], traj_array[:, 1],
                    color='lightblue', alpha=0.3, linewidth=1, zorder=2
                )

        # 绘制当前轨迹
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax_env.plot(
                traj[:, 0], traj[:, 1],
                color='red', alpha=0.8, linewidth=2.5,
                label='Current Path', zorder=8
            )

            # 绘制起点
            self.ax_env.plot(
                traj[0, 0], traj[0, 1],
                marker='o', markersize=10, color='orange',
                markeredgecolor='black', markeredgewidth=1.5,
                label='Start', zorder=9
            )

        # 绘制目标点（带光晕效果）
        goal_glow = Circle(
            (self.env.goal_x, self.env.goal_y),
            0.35,
            color='lime',
            alpha=0.3,
            zorder=7
        )
        self.ax_env.add_patch(goal_glow)

        goal = Circle(
            (self.env.goal_x, self.env.goal_y),
            0.2,
            # color='green',
            facecolor='green',
            alpha=0.9,
            edgecolor='darkgreen',
            linewidth=2,
            label='Goal',
            zorder=10
        )
        self.ax_env.add_patch(goal)

        # 绘制机器人（带朝向箭头）
        robot_body = Circle(
            (self. env.x, self.env.y),
            0.25,
            # color='dodgerblue',
            facecolor='dodgerblue',
            alpha=0.9,
            edgecolor='darkblue',
            linewidth=2,
            zorder=15
        )
        self.ax_env.add_patch(robot_body)

        # 机器人朝向箭头
        arrow_length = 0.4
        dx = arrow_length * np.cos(self.env.theta)
        dy = arrow_length * np.sin(self.env.theta)
        self.ax_env.arrow(
            self.env.x, self.env.y, dx, dy,
            head_width=0.2, head_length=0.15,
            fc='yellow', ec='orange', linewidth=2,
            zorder=16
        )

        # 绘制激光扫描线（部分）
        laser_data = obs['laser']
        angles = np.linspace(-np.pi/2, np. pi/2, len(laser_data))

        # 只绘制每5个激光束
        for i in range(0, len(laser_data), 5):
            if laser_data[i] < self.env.laser_range:
                angle = self.env.theta + angles[i]
                end_x = self.env.x + laser_data[i] * np.cos(angle)
                end_y = self.env. y + laser_data[i] * np.sin(angle)

                # 根据距离设置颜色
                if laser_data[i] < 0.5:
                    color = 'red'
                    alpha = 0.6
                elif laser_data[i] < 1.0:
                    color = 'orange'
                    alpha = 0.4
                else:
                    color = 'cyan'
                    alpha = 0.2

                self. ax_env.plot(
                    [self.env. x, end_x], [self.env.y, end_y],
                    color=color, alpha=alpha, linewidth=0.5, zorder=3
                )

        # 绘制到目标的连线
        self.ax_env.plot(
            [self.env.x, self.env.goal_x],
            [self.env. y, self.env.goal_y],
            'g--', alpha=0.4, linewidth=1.5, zorder=4
        )

        # 添加信息文本框
        distance = obs['robot_state'][0]
        angle_to_goal = np.degrees(obs['robot_state'][1])

        info_text = (
            f'Episode: {self.episode_count}\n'
            f'Step: {self.step_count}\n'
            f'Distance: {distance:.2f}m\n'
            f'Angle: {angle_to_goal:.1f}\n'
            f'Reward: {reward:.2f}\n'
            f'Total: {self.total_reward:.1f}'
        )

        # 创建文本框
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        self.ax_env.text(
            0.02, 0.98, info_text,
            transform=self.ax_env.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props,
            fontfamily='monospace'
        )

        # 添加图例
        self.ax_env.legend(loc='upper right', fontsize=9, framealpha=0.9)

        # 设置标题
        status = "🟢 Active" if distance > 0.3 else "🎯 Goal Reached!"
        self.ax_env. set_title(
            f'Navigation Environment - {status}',
            fontsize=12, fontweight='bold'
        )

    def _draw_lidar(self, obs):
        """绘制激光雷达数据（极坐标）"""
        self.ax_laser.clear()

        laser_data = obs['laser']
        angles = np.linspace(-np.pi/2, np. pi/2, len(laser_data))

        # 绘制激光数据
        self.ax_laser.plot(angles, laser_data, 'b-', linewidth=2, label='Scan')
        self.ax_laser.fill(angles, laser_data, 'blue', alpha=0.3)

        # 标记危险区域
        danger_threshold = 0.6
        danger_indices = np.where(np.array(laser_data) < danger_threshold)[0]
        if len(danger_indices) > 0:
            danger_angles = angles[danger_indices]
            danger_dists = np.array(laser_data)[danger_indices]
            self.ax_laser.scatter(
                danger_angles, danger_dists,
                color='red', s=50, zorder=10,
                label='Danger Zone', alpha=0.8
            )

        self.ax_laser.set_ylim(0, self.env. laser_range)
        self. ax_laser.set_title('Lidar Scan', fontsize=10, fontweight='bold', pad=20)
        self.ax_laser.legend(loc='upper right', fontsize=8)
        self.ax_laser.grid(True, alpha=0.3)

    # 在 _draw_action_history 方法中
    def _draw_action_history(self, action):
        """绘制动作历史"""
        self.ax_action.clear()

        self.action_history.append(action)

        if len(self.action_history) > 1:
            history = np.array(list(self.action_history))
            steps = np.arange(len(history))

            # 绘制线速度
            self.ax_action.plot(
                steps, history[:, 0],
                'b-', linewidth=2, label='Linear Vel', marker='o', markersize=3
            )

            # 绘制角速度
            self.ax_action.plot(
                steps, history[:, 1],
                'r-', linewidth=2, label='Angular Vel', marker='s', markersize=3
            )

            # 填充区域
            self.ax_action.fill_between(
                steps, history[:, 0], alpha=0.3, color='blue'
            )
            self.ax_action.fill_between(
                steps, history[:, 1], alpha=0.3, color='red'
            )

            # 只在有数据时添加图例
            self.ax_action.legend(loc='upper right', fontsize=8)  # ← 移到这里

        self.ax_action.set_ylim(-1.2, 1.2)
        self.ax_action.set_xlim(0, 50)
        self.ax_action.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_action.set_xlabel('Time Steps', fontsize=9)
        self.ax_action.set_ylabel('Action Value', fontsize=9)
        self.ax_action.set_title('Action History', fontsize=10, fontweight='bold')
        # self.ax_action.legend(loc='upper right', fontsize=8)  # ← 删除这行
        self.ax_action.grid(True, alpha=0.3)

    def _draw_reward_history(self, reward):  # ← 添加这个方法
        """绘制奖励历史"""
        self.ax_reward.clear()

        self.reward_history.append(reward)

        if len(self.reward_history) > 1:
            rewards = np.array(list(self.reward_history))
            steps = np.arange(len(rewards))

            # 绘制奖励曲线
            self.ax_reward.plot(
                steps, rewards,
                'g-', linewidth=2, alpha=0.8
            )

            # 填充正奖励（绿色）和负奖励（红色）
            self.ax_reward.fill_between(
                steps, 0, rewards,
                where=(rewards >= 0),
                color='green', alpha=0.3, label='Positive'
            )
            self.ax_reward.fill_between(
                steps, 0, rewards,
                where=(rewards < 0),
                color='red', alpha=0.3, label='Negative'
            )

            # 绘制移动平均
            if len(rewards) >= 10:
                window = 10
                moving_avg = np.convolve(
                    rewards, np.ones(window) / window, mode='valid'
                )
                avg_steps = steps[window - 1:]
                self.ax_reward.plot(
                    avg_steps, moving_avg,
                    'k--', linewidth=2, label='Moving Avg', alpha=0.7
                )

            # 只在有数据时添加图例
            self.ax_reward.legend(loc='upper right', fontsize=8)

        # 设置坐标轴和标签
        self.ax_reward.set_xlim(0, 200)
        self.ax_reward.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_reward.set_xlabel('Steps', fontsize=9)
        self.ax_reward.set_ylabel('Reward', fontsize=9)
        self.ax_reward.set_title('Reward History', fontsize=10, fontweight='bold')
        self.ax_reward.grid(True, alpha=0.3)

    def reset(self):
        """重置可视化器"""
        if len(self.trajectory) > 0:
            self.all_trajectories.append(self. trajectory.copy())

        self.trajectory = []
        self. reward_history. clear()
        self.action_history.clear()
        self.step_count = 0
        self.total_reward = 0
        self.episode_count += 1

    def update(self, obs, action, reward):
        """
        更新可视化

        Args:
            obs: 环境观测
            action: 执行的动作
            reward: 获得的奖励
        """
        # 记录轨迹
        self.trajectory.append([self. env.x, self.env. y])

        # 更新统计
        self. step_count += 1
        self.total_reward += reward

        # 绘制所有子图
        self._draw_environment(obs, action, reward)
        self._draw_lidar(obs)
        self._draw_action_history(action)
        self._draw_reward_history(reward)

        # 刷新显示
        plt.pause(0.001)

    def show(self):
        """显示图形"""
        plt.show()

    def save_figure(self, filename='td3_visualization. png'):
        """保存当前图形"""
        self. fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"图形已保存至: {filename}")


# ============================================================
# 测试代码（必须在类定义之外）
# ============================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os. path.join(os.path.dirname(__file__), '.. ', '..')))

    from gdae_td3.src.environment.simulator import RobotSimulator
    from gdae_td3.src.td3.agent import TD3Agent
    import torch

    print("=" * 60)
    print("测试 TD3Visualizer")
    print("=" * 60)

    # 创建环境和智能体
    print("\n创建环境...")
    env = RobotSimulator()

    print("创建智能体...")
    agent = TD3Agent(device=torch.device('cpu'))

    # 创建可视化器
    print("创建可视化器...")
    visualizer = TD3Visualizer(env, agent)

    # 运行测试
    print("开始测试导航.. .\n")
    obs = env.reset()

    for step in range(100):
        # 简单的启发式动作（朝向目标）
        distance, angle = obs['robot_state']
        min_laser = min(obs['laser'])

        # 避障逻辑
        if min_laser < 0.5:
            # 转向更空旷的方向
            left_avg = np.mean(obs['laser'][:len(obs['laser'])//2])
            right_avg = np.mean(obs['laser'][len(obs['laser'])//2:])
            action = [0.1, 0.8 if left_avg > right_avg else -0.8]
        else:
            # 朝向目标
            linear_vel = min(0.4, distance / 2.0)
            angular_vel = np.clip(angle * 2.0, -0.5, 0.5)
            action = [linear_vel, angular_vel]

        action_in = [(action[0] + 1) / 2, action[1]]
        next_obs, reward, done, info = env.step(action_in)

        # 更新可视化
        visualizer.update(obs, action, reward)

        if done:
            print(f"\nEpisode 结束于第 {step+1} 步")
            if info.get('distance_to_goal', 1.0) < 0.3:
                print("✓ 成功到达目标！")
            elif info.get('collision', False):
                print("✗ 发生碰撞")
            break

        obs = next_obs

    print("\n测试完成！关闭窗口以退出...")
    visualizer.show()