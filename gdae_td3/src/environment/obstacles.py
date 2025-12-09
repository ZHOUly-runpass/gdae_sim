import numpy as np


class ObstacleManager:
    """障碍物管理器"""

    def __init__(self, map_size):
        """
        初始化障碍物管理器

        Args:
            map_size: 地图尺寸
        """
        self.map_size = map_size
        self.obstacles = []
        self.robot_radius = 0.2  # 机器人半径

    def reset(self, num_obstacles=10):
        """
        随机生成障碍物

        Args:
            num_obstacles: 障碍物数量
        """
        self.obstacles = []
        half_size = self.map_size / 2

        for _ in range(num_obstacles):
            # 在地图内部生成障碍物（留出边缘空间）
            x = np.random.uniform(-half_size + 1.0, half_size - 1.0)
            y = np.random.uniform(-half_size + 1.0, half_size - 1.0)
            radius = np.random.uniform(0.3, 0.8)  # 障碍物半径
            self.obstacles.append({'x': x, 'y': y, 'radius': radius})

    def check_collision(self, x, y, robot_radius=None):
        """
        检查是否与障碍物或边界发生碰撞

        Args:
            x: 机器人 x 坐标
            y: 机器人 y 坐标
            robot_radius: 机器人半径（可选）

        Returns:
            bool: 是否发生碰撞
        """
        if robot_radius is None:
            robot_radius = self.robot_radius

        half_size = self.map_size / 2

        # 1. 检查边界碰撞
        if (x < -half_size + robot_radius or
                x > half_size - robot_radius or
                y < -half_size + robot_radius or
                y > half_size - robot_radius):
            return True

        # 2. 检查障碍物碰撞
        for obs in self.obstacles:
            dist = np.sqrt((x - obs['x']) ** 2 + (y - obs['y']) ** 2)
            if dist < obs['radius'] + robot_radius:
                return True

        return False

    def get_obstacles(self):
        """返回所有障碍物列表"""
        return self.obstacles