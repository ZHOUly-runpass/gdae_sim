import numpy as np

class ObstacleManager:
    def __init__(self, map_size):
        self.map_size = map_size
        self.obstacles = []

    def reset(self, num_obstacles=10):
        """随机生成障碍物"""
        self.obstacles = []
        for _ in range(num_obstacles):
            x = np.random.uniform(-self.map_size / 2, self.map_size / 2)
            y = np.random.uniform(-self.map_size / 2, self.map_size / 2)
            radius = np.random.uniform(0.2, 0.5)  # 障碍物大小
            self.obstacles.append({'x': x, 'y': y, 'radius': radius})

    def check_collision(self, x, y):
        """检查是否与障碍物发生碰撞"""
        for obs in self.obstacles:
            if np.linalg.norm([x - obs['x'], y - obs['y']]) < obs['radius'] + 0.2:
                return True
        return False