import numpy as np
import random
import math

from .obstacles import ObstacleManager
from .sensors import LidarSensor


class RobotSimulator:
    """
    TD3 局部导航仿真器
    负责机器人运动控制、激光数据生成和障碍物碰撞检测
    """

    def __init__(self, map_size=10.0, laser_range=5.0, laser_dim=20, velocity_limits=(0.5, 1.0)):
        # 地图尺寸（正方形 map_size x map_size）
        self.map_size = map_size

        # 激光雷达配置
        self.laser_range = laser_range
        self.laser_dim = laser_dim
        self.lidar = LidarSensor(map_size, laser_range, laser_dim)

        # 机器人运动学配置
        self.max_linear_velocity, self.max_angular_velocity = velocity_limits
        self.x, self.y, self.theta = 0.0, 0.0, 0.0  # 初始状态

        # 障碍物管理
        self.obstacles = ObstacleManager(map_size)
        self.obstacles.reset()  # 随机生成障碍物

        # 目标点
        self.goal_x, self.goal_y = 0.0, 0.0
        self.goal_reach_threshold = 0.3

    def reset(self):
        """重置环境及机器人状态"""
        # 设置机器人初始位置
        while True:
            self.x = random.uniform(-self.map_size / 2, self.map_size / 2)
            self.y = random.uniform(-self.map_size / 2, self.map_size / 2)
            if not self.obstacles.check_collision(self.x, self.y):
                break
        self.theta = random.uniform(-math.pi, math.pi)

        # 设置随机目标点
        while True:
            self.goal_x = random.uniform(-self.map_size / 2, self.map_size / 2)
            self.goal_y = random.uniform(-self.map_size / 2, self.map_size / 2)
            if not self.obstacles.check_collision(self.goal_x, self.goal_y):
                break

        return self.get_observation()

    def step(self, action):
        """
        执行动作并推进仿真环境一步
        :param action: [linear_velocity, angular_velocity]
        :return:
            obs: 当前观测（激光数据、目标距离角度）
            reward: 当前步奖励
            done: 是否结束
            info: 额外信息
        """
        # 执行动作
        linear_vel, angular_vel = action
        self.x += linear_vel * math.cos(self.theta) * 0.1
        self.y += linear_vel * math.sin(self.theta) * 0.1
        self.theta += angular_vel * 0.1
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))  # 角度归一化

        # 检测目标和碰撞
        distance_to_goal = np.linalg.norm([self.goal_x - self.x, self.goal_y - self.y])
        collision = self.obstacles.check_collision(self.x, self.y)

        # 奖励函数
        reward = self.compute_reward(distance_to_goal, collision, action)

        # 环境结束条件
        done = collision or distance_to_goal < self.goal_reach_threshold

        return self.get_observation(), reward, done, {
            'distance_to_goal': distance_to_goal,
            'collision': collision
        }

    def get_observation(self):
        """返回当前状态观测"""
        # 获取激光雷达数据
        laser_data = self.lidar.get_lidar_data(self.x, self.y, self.theta, self.obstacles)

        # 计算目标的相对方向
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        distance_to_goal = np.linalg.norm([dx, dy])
        relative_angle = math.atan2(dy, dx) - self.theta
        relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))

        return {
            'laser': laser_data,
            'robot_state': [distance_to_goal, relative_angle],
        }

    def compute_reward(self, distance, collision, action):
        """计算奖励函数"""
        if collision:  # 碰撞惩罚
            return -100.0
        elif distance < self.goal_reach_threshold:  # 到达目标奖励
            return 100.0
        else:  # 综合式奖励
            linear_reward = action[0] * 0.5  # 线速度奖励
            angular_penalty = -abs(action[1]) * 0.2  # 转向惩罚
            obstacle_penalty = -min(1.0, (1.0 / distance - 0.1))  # 距离障碍物的惩罚
            return linear_reward + angular_penalty + obstacle_penalty

    def render(self):
        """环境可视化"""
        # 可视化函数（结合 matplotlib、pygame 实现）
        pass