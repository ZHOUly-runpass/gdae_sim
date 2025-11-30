import numpy as np
import math

class LidarSensor:
    def __init__(self, map_size, range, dim):
        self.map_size = map_size
        self.range = range
        self.dim = dim
        self.angles = np.linspace(-math.pi / 2, math.pi / 2, dim)

    def get_lidar_data(self, x, y, theta, obstacle_manager):
        """模拟激光扫描"""
        distances = []
        for angle in self.angles:
            laser_angle = theta + angle
            dx, dy = math.cos(laser_angle), math.sin(laser_angle)
            min_distance = self.range

            for obs in obstacle_manager.obstacles:
                dist_to_obs = self.ray_circle_intersection(x, y, dx, dy, obs)
                if dist_to_obs is not None and dist_to_obs < min_distance:
                    min_distance = dist_to_obs

            distances.append(min_distance)
        return distances

    def ray_circle_intersection(self, x, y, dx, dy, obstacle):
        """计算激光与圆形障碍物的交点"""
        cx, cy, r = obstacle['x'], obstacle['y'], obstacle['radius']
        a = dx**2 + dy**2
        b = 2 * (dx * (x - cx) + dy * (y - cy))
        c = (x - cx)**2 + (y - cy)**2 - r**2
        delta = b**2 - 4 * a * c
        if delta < 0:
            return None
        sqrt_delta = math.sqrt(delta)
        t1, t2 = (-b - sqrt_delta) / (2 * a), (-b + sqrt_delta) / (2 * a)
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None