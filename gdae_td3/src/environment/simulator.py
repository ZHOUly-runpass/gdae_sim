import numpy as np
import random
import math

from . obstacles import ObstacleManager
from .sensors import LidarSensor


class RobotSimulator:
    """
    TD3 局部导航仿真器
    负责机器人运动控制、激光数据生成和障碍物碰撞检测
    """

    def __init__(self, map_size=10.0, laser_range=5.0, laser_dim=20,
                 velocity_limits=(0.5, 2.0), time_step=0.1):
        """
        初始化仿真环境
        """
        self.map_size = map_size
        self.laser_range = laser_range
        self.laser_dim = laser_dim
        self.lidar = LidarSensor(map_size, laser_range, laser_dim)

        self.max_linear_velocity, self.max_angular_velocity = velocity_limits
        self.max_linear_vel = self.max_linear_velocity
        self.max_angular_vel = self.max_angular_velocity
        self.time_step = time_step

        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        self.obstacles = ObstacleManager(map_size)
        self.obstacles.reset()

        self.goal_x, self.goal_y = 0.0, 0.0
        self.goal_reach_threshold = 0.3
        self.collision_dist = 0.35

        # 保存当前执行的动作（用于状态构建，范围 [-1, 1]）
        self.current_action = np.array([0.0, 0.0])

    def reset(self):
        """重置环境及机器人状态"""
        self.obstacles.reset()

        while True:
            self.x = random.uniform(-self.map_size / 2, self.map_size / 2)
            self.y = random.uniform(-self.map_size / 2, self.map_size / 2)
            if not self.obstacles.check_collision(self.x, self.y):
                break
        self.theta = random.uniform(-math.pi, math.pi)

        while True:
            self.goal_x = random.uniform(-self.map_size / 2, self.map_size / 2)
            self.goal_y = random.uniform(-self.map_size / 2, self.map_size / 2)
            if not self.obstacles.check_collision(self.goal_x, self.goal_y):
                break

        # 重置动作
        self.current_action = np.array([0.0, 0.0])

        return self.get_observation()

    def step(self, action):
        """
        执行动作并推进仿真环境一步

        Args:
            action: [linear_velocity, angular_velocity]
                   linear_velocity: [0, 1] 范围
                   angular_velocity: [-1, 1] 范围
        """
        linear_vel, angular_vel = action

        # 缩放到实际速度范围
        actual_linear_vel = linear_vel * self.max_linear_vel
        actual_angular_vel = angular_vel * self.max_angular_vel

        self.current_linear_vel = actual_linear_vel
        self. current_angular_vel = actual_angular_vel

        # 更新位置和朝向
        self.x += actual_linear_vel * math.cos(self.theta) * self.time_step
        self.y += actual_linear_vel * math.sin(self. theta) * self.time_step
        self.theta += actual_angular_vel * self.time_step

        # 保存当前动作（转换回网络输出范围 [-1, 1]）
        # action_in = [(network_action[0] + 1) / 2, network_action[1]]
        # 所以 network_action[0] = action[0] * 2 - 1
        self.current_action = np.array([action[0] * 2 - 1, action[1]])

        # 获取激光数据用于碰撞检测
        laser_data = self.lidar. get_lidar_data(self.x, self.y, self.theta, self.obstacles)
        min_laser = min(laser_data)

        # 检测碰撞
        collision = min_laser < self.collision_dist

        # 检测目标
        distance_to_goal = np.linalg.norm([self.goal_x - self. x, self.goal_y - self.y])
        reach_goal = distance_to_goal < self.goal_reach_threshold

        # 计算奖励
        reward = self.compute_reward(reach_goal, collision, action, min_laser)

        done = collision or reach_goal

        return self.get_observation(), reward, done, {
            'distance_to_goal': distance_to_goal,
            'collision':  collision,
            'reach_goal': reach_goal,
            'min_laser': min_laser
        }

    def get_observation(self):
        """
        返回当前状态观测
        """
        laser_data = self.lidar. get_lidar_data(self.x, self.y, self.theta, self.obstacles)

        # 计算目标的相对方向
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        distance_to_goal = np.linalg.norm([dx, dy])

        # 计算相对角度
        dot = dx * 1 + dy * 0
        mag1 = math.sqrt(dx ** 2 + dy ** 2)

        if mag1 > 0:
            beta = math.acos(np.clip(dot / mag1, -1.0, 1.0))
            if dy < 0:
                beta = -beta
        else:
            beta = 0.0

        theta = beta - self.theta

        # 归一化角度到 [-pi, pi]
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        return {
            'laser':  laser_data,
            'robot_state': np.array([distance_to_goal, theta]),
            'action': self.current_action. copy()
        }

    def compute_reward(self, target, collision, action, min_laser):
        """
        严格遵循 DRL-robot-navigation 的奖励函数
        """
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            # 转换动作到网络输出范围
            network_action_0 = action[0] * 2 - 1  # [0,1] → [-1,1]
            network_action_1 = action[1]  # 已经是 [-1,1]

            # 归一化激光数据到 [0, 1] 范围
            normalized_laser = min_laser / self.laser_range  # 除以 5.0

            # 障碍物惩罚函数
            r3 = lambda x: 1 - x if x < 1 else 0.0

            # 简单的动作奖励
            reward = (network_action_0 / 2 -
                      abs(network_action_1) / 2 -
                      r3(normalized_laser) / 2)  # 使用归一化后的激光值

            return reward

    def render(self):
        """环境可视化"""
        pass