"""
测试激光雷达传感器模块
"""
import sys
import os
# 添加 src 目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import math
from gdae_td3.src.environment.sensors import LidarSensor
from gdae_td3.src.environment.obstacles import ObstacleManager


def test_lidar_initialization():
    """测试激光雷达初始化"""
    print("\n" + "=" * 60)
    print("测试 1: 激光雷达初始化")
    print("=" * 60)

    map_size = 10.0
    laser_range = 5.0
    laser_dim = 20

    lidar = LidarSensor(map_size, laser_range, laser_dim)

    assert lidar.map_size == map_size, "地图尺寸不匹配"
    assert lidar.range == laser_range, "激光范围不匹配"
    assert lidar.dim == laser_dim, "激光维度不匹配"
    assert len(lidar.angles) == laser_dim, "激光角度数组长度不正确"

    print(f"✓ 地图尺寸: {lidar.map_size}")
    print(f"✓ 激光范围: {lidar.range}m")
    print(f"✓ 激光维度: {lidar.dim}")
    print(f"✓ 角度范围: [{math.degrees(lidar.angles[0]):.1f}°, {math.degrees(lidar.angles[-1]):.1f}°]")
    print("✓ 激光雷达初始化成功")


def test_lidar_no_obstacles():
    """测试无障碍物环境的激光扫描"""
    print("\n" + "=" * 60)
    print("测试 2: 无障碍物激光扫描")
    print("=" * 60)

    lidar = LidarSensor(10.0, 5.0, 20)
    obs_manager = ObstacleManager(10.0)
    obs_manager.obstacles = []  # 无障碍物

    x, y, theta = 0.0, 0.0, 0.0
    laser_data = lidar.get_lidar_data(x, y, theta, obs_manager)

    assert len(laser_data) == 20, "激光数据维度不正确"
    assert all(d == 5.0 for d in laser_data), "无障碍物时所有距离应为最大值"

    print(f"✓ 激光数据维度: {len(laser_data)}")
    print(f"✓ 所有距离均为最大值: {all(d == 5.0 for d in laser_data)}")
    print(f"  示例数据: {laser_data[:5]}")
    print("✓ 无障碍物扫描测试通过")


def test_lidar_with_single_obstacle():
    """测试单个障碍物的激光扫描"""
    print("\n" + "=" * 60)
    print("测试 3: 单障碍物激光扫描")
    print("=" * 60)

    lidar = LidarSensor(10.0, 5.0, 20)
    obs_manager = ObstacleManager(10.0)

    # 在机器人正前方放置一个障碍物
    obs_manager.obstacles = [{'x': 2.0, 'y': 0.0, 'radius': 0.5}]

    x, y, theta = 0.0, 0.0, 0.0
    laser_data = lidar.get_lidar_data(x, y, theta, obs_manager)

    # 中间的激光束应该检测到障碍物
    middle_index = len(laser_data) // 2
    middle_distance = laser_data[middle_index]

    print(f"✓ 激光数据维度: {len(laser_data)}")
    print(f"✓ 正前方距离: {middle_distance:.2f}m (预期约 1.5m)")
    print(f"  完整激光数据:")
    for i, dist in enumerate(laser_data):
        angle_deg = math.degrees(lidar.angles[i])
        print(f"    [{i:2d}] 角度={angle_deg:6.1f}°, 距离={dist:.2f}m")

    # 正前方应该检测到障碍物（距离约为 2.0 - 0.5 = 1.5m）
    assert 1.0 < middle_distance < 2.0, f"正前方距离异常: {middle_distance}"

    # 两侧应该没有障碍物
    assert laser_data[0] == 5.0, "左侧应该无障碍物"
    assert laser_data[-1] == 5.0, "右侧应该无障碍物"

    print("✓ 单障碍物扫描测试通过")


def test_lidar_with_multiple_obstacles():
    """测试多障碍物的激光扫描"""
    print("\n" + "=" * 60)
    print("测试 4: 多障碍物激光扫描")
    print("=" * 60)

    lidar = LidarSensor(10.0, 5.0, 20)
    obs_manager = ObstacleManager(10.0)

    # 放置多个障碍物
    obs_manager.obstacles = [
        {'x': 2.0, 'y': 0.0, 'radius': 0.3},  # 正前方
        {'x': 1.5, 'y': 1.5, 'radius': 0.3},  # 左前方
        {'x': 1.5, 'y': -1.5, 'radius': 0.3}  # 右前方
    ]

    x, y, theta = 0.0, 0.0, 0.0
    laser_data = lidar.get_lidar_data(x, y, theta, obs_manager)

    # 检查是否检测到多个障碍物
    detected_obstacles = sum(1 for d in laser_data if d < 5.0)

    print(f"✓ 检测到 {detected_obstacles} 个方向有障碍物")
    print(f"✓ 最小距离: {min(laser_data):.2f}m")
    print(f"✓ 最大距离: {max(laser_data):.2f}m")
    print(f"  激光数据统计:")
    print(f"    < 2m: {sum(1 for d in laser_data if d < 2.0)} 个方向")
    print(f"    2-3m: {sum(1 for d in laser_data if 2.0 <= d < 3.0)} 个方向")
    print(f"    > 3m: {sum(1 for d in laser_data if d >= 3.0)} 个方向")

    assert detected_obstacles > 0, "应该检测到至少一个障碍物"
    print("✓ 多障碍物扫描测试通过")


def test_ray_circle_intersection():
    """测试射线-圆形相交算法"""
    print("\n" + "=" * 60)
    print("测试 5: 射线-圆形相交算法")
    print("=" * 60)

    lidar = LidarSensor(10.0, 5.0, 20)

    test_cases = [
        # (射线起点x, y, 方向dx, dy, 障碍物, 是否相交, 描述)
        (0.0, 0.0, 1.0, 0.0, {'x': 2.0, 'y': 0.0, 'radius': 0.5}, True, "射线直接命中圆心"),
        (0.0, 0.0, 0.0, 1.0, {'x': 2.0, 'y': 0.0, 'radius': 0.5}, False, "射线与圆不相交"),
        (0.0, 0.0, 1.0, 0.0, {'x': -2.0, 'y': 0.0, 'radius': 0.5}, False, "射线反向，不相交"),
        (0.0, 0.0, 1.0, 0.1, {'x': 2.0, 'y': 0.2, 'radius': 0.5}, True, "射线斜向命中"),
    ]

    for x, y, dx, dy, obs, should_hit, description in test_cases:
        result = lidar.ray_circle_intersection(x, y, dx, dy, obs)
        hit = result is not None
        status = "✓" if hit == should_hit else "✗"

        if result:
            print(f"{status} {description}: 相交距离={result:.2f}m")
        else:
            print(f"{status} {description}: 不相交")

        assert hit == should_hit, f"相交检测失败: {description}"

    print("✓ 射线-圆形相交算法测试通过")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试激光雷达传感器模块")
    print("=" * 60)

    try:
        test_lidar_initialization()
        test_lidar_no_obstacles()
        test_lidar_with_single_obstacle()
        test_lidar_with_multiple_obstacles()
        test_ray_circle_intersection()

        print("\n" + "=" * 60)
        print("✓ 所有激光雷达测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        sys.exit(1)