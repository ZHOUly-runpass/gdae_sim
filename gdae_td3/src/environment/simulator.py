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

    def __init__(self, map_size=10.0, laser_range=5.0, laser_dim=20,
                 velocity_limits=(0.5, 2.0), time_step=0.1):
        """
        初始化仿真环境

        Args:
            map_size: 地图尺寸（正方形 map_size x map_size）
            laser_range: 激光雷达最大探测距离
            laser_dim: 激光雷达分辨率（光束数量）
            velocity_limits: (max_linear_velocity, max_angular_velocity)
                            默认 (0.5 m/s, 2.0 rad/s)
            time_step: 仿真时间步长（秒），默认 0.1s
        """
        # 地图尺寸（正方形 map_size x map_size）
        self.map_size = map_size

        # 激光雷达配置
        self.laser_range = laser_range
        self. laser_dim = laser_dim
        self.lidar = LidarSensor(map_size, laser_range, laser_dim)

        # 机器人运动学配置
        self. max_linear_velocity, self. max_angular_velocity = velocity_limits
        # 添加别名以兼容 step() 方法中的命名
        self.max_linear_vel = self.max_linear_velocity
        self.max_angular_vel = self.max_angular_velocity
        # 添加时间步长
        self.time_step = time_step

        # 机器人状态
        self.x, self. y, self.theta = 0.0, 0.0, 0.0

        # 障碍物管理
        self.obstacles = ObstacleManager(map_size)
        self.obstacles.reset()  # 随机生成障碍物

        # 目标点
        self.goal_x, self.goal_y = 0.0, 0.0
        self.goal_reach_threshold = 0.3

    def reset(self):
        """
        重置环境及机器人状态
        """
        # 重新生成障碍物
        self.obstacles.reset()

        # 设置机器人初始位置（避免与障碍物重叠）
        while True:
            self.x = random.uniform(-self.map_size / 2, self.map_size / 2)
            self.y = random.uniform(-self.map_size / 2, self.map_size / 2)
            if not self.obstacles.check_collision(self.x, self.y):
                break
        self.theta = random.uniform(-math.pi, math.pi)

        # 设置随机目标点（避免与障碍物重叠）
        while True:
            self.goal_x = random.uniform(-self.map_size / 2, self.map_size / 2)
            self.goal_y = random.uniform(-self.map_size / 2, self.map_size / 2)
            if not self.obstacles.check_collision(self.goal_x, self.goal_y):
                break

        # 获取当前环境观测
        observation = self.get_observation()

        # 初始化 last_distance（当前机器人与目标的距离）
        self.last_distance = observation['robot_state'][0]  # 距目标的初始距离

        return observation


    def step(self, action):
        """
        执行动作并推进仿真环境一步

        Args:
            action: [linear_velocity, angular_velocity]
                   linear_velocity: [0, 1] 范围，会被缩放到 [0, max_linear_velocity]
                   angular_velocity: [-1, 1] 范围，会被缩放到 [-max_angular_velocity, max_angular_velocity]

        Returns:
            obs: 当前观测（激光数据、目标距离角度）
            reward: 当前步奖励
            done: 是否结束
            info: 额外信息字典
        """
        # 执行动作
        linear_vel, angular_vel = action

        # 缩放到实际速度范围
        actual_linear_vel = linear_vel * self.max_linear_vel
        actual_angular_vel = angular_vel * self.max_angular_vel

        # 更新位置和朝向
        self.x += actual_linear_vel * math.cos(self.theta) * self.time_step
        self.y += actual_linear_vel * math.sin(self. theta) * self.time_step
        self.theta += actual_angular_vel * self.time_step
        self.theta = math. atan2(math.sin(self.theta), math.cos(self.theta))  # 角度归一化

        # 检测目标和碰撞
        distance_to_goal = np.linalg.norm([self.goal_x - self. x, self.goal_y - self.y])
        collision = self.obstacles.check_collision(self.x, self.y)

        # 判断是否到达目标
        reach_goal = distance_to_goal < self.goal_reach_threshold

        # 奖励函数
        reward = self.compute_reward(distance_to_goal, collision, action)

        # 环境结束条件
        done = collision or reach_goal

        return self.get_observation(), reward, done, {
            'distance_to_goal': distance_to_goal,
            'collision': collision,
            'reach_goal': reach_goal
        }

    def get_observation(self):
        """返回当前状态观测"""
        # 获取激光雷达数据
        laser_data = self.lidar. get_lidar_data(self.x, self.y, self.theta, self.obstacles)

        # 计算目标的相对方向
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        distance_to_goal = np.linalg.norm([dx, dy])
        relative_angle = math.atan2(dy, dx) - self.theta
        relative_angle = math. atan2(math.sin(relative_angle), math.cos(relative_angle))

        return {
            'laser': laser_data,
            'robot_state': [distance_to_goal, relative_angle],
        }

    def compute_reward(self, distance, collision, action):
        """
        计算奖励函数

        Args:
            distance: 到目标的距离
            collision: 是否碰撞
            action: [linear_vel, angular_vel]，范围 [0,1] 和 [-1,1]
        """
        if collision:
            return -100.0
        elif distance < self.goal_reach_threshold:
            return 100.0
        else:
            # 1. 前进奖励
            linear_reward = action[0] * 0.3

            # 2. 朝向目标奖励（新增）
            dx, dy = self.goal_x - self.x, self.goal_y - self.y
            angle_to_goal = math.atan2(dy, dx)
            angle_diff = abs(math.atan2(
                math.sin(angle_to_goal - self.theta),
                math.cos(angle_to_goal - self.theta)
            ))
            # 朝向越准确，奖励越大
            heading_reward = (math.pi - angle_diff) / math.pi * 0.5

            # 3. 距离减小奖励（新增）
            if hasattr(self, 'last_distance'):
                distance_reward = (self.last_distance - distance) * 2.0
            else:
                distance_reward = 0.0
            self.last_distance = distance

            # 4. 转向惩罚（大幅降低）
            angular_penalty = -abs(action[1]) * 0.01

            # 5.  障碍物惩罚（修复）
            laser_data = self.lidar.get_lidar_data(
                self.x, self.y, self.theta, self.obstacles
            )
            min_laser = min(laser_data)
            if min_laser < 0.5:
                obstacle_penalty = -(0.5 - min_laser) * 1.0
            else:
                obstacle_penalty = 0.0

            total_reward = (
                    linear_reward +
                    heading_reward +
                    distance_reward +
                    angular_penalty +
                    obstacle_penalty
            )

            return total_reward

    def render(self):
        """环境可视化"""
        pass