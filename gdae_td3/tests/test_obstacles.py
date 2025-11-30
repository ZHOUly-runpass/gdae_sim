"""
测试障碍物管理模块
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '. .')))

import numpy as np
from gdae_td3.src.environment.obstacles import ObstacleManager


def test_obstacle_creation():
    """测试障碍物生成"""
    print("\n" + "=" * 60)
    print("测试 1: 障碍物生成")
    print("=" * 60)

    map_size = 10.
    0
    obs_manager = ObstacleManager(map_size)
    obs_manager.reset(num_obstacles=10)

    assert len(obs_manager.obstacles) == 10, "障碍物数量不正确"
    print(f"✓ 成功生成 {len(obs_manager.obstacles)} 个障碍物")

    # 检查障碍物是否在地图范围内
    for i, obs in enumerate(obs_manager.obstacles):
        assert -map_size / 2 <= obs['x'] <= map_size / 2, f"障碍物 {i} X 坐标超出范围"
        assert -map_size / 2 <= obs['y'] <= map_size / 2, f"障碍物 {i} Y 坐标超出范围"
        assert 0.2 <= obs['radius'] <= 0.5, f"障碍物 {i} 半径不在合理范围"
        print(f"  障碍物 {i}: 位置=({obs['x']:.2f}, {obs['y']:.2f}), 半径={obs['radius']:.2f}")

    print("✓ 所有障碍物参数合法")


def test_collision_detection():
    """测试碰撞检测"""
    print("\n" + "=" * 60)
    print("测试 2: 碰撞检测")
    print("=" * 60)

    obs_manager = ObstacleManager(10.0)
    obs_manager.obstacles = [
        {'x': 0.0, 'y': 0.0, 'radius': 0.5},
        {'x': 3.0, 'y': 3.0, 'radius': 0.3}
    ]

    # 测试碰撞情况
    collision_cases = [
        (0.0, 0.0, True, "机器人在障碍物中心"),
        (0.6, 0.0, True, "机器人与障碍物边缘接触"),
        (1.0, 0.0, False, "机器人远离障碍物"),
        (3.0, 3.0, True, "机器人在第二个障碍物中"),
        (5.0, 5.0, False, "机器人在安全区域")
    ]

    for x, y, expected, description in collision_cases:
        result = obs_manager.check_collision(x, y)
        status = "✓" if result == expected else "✗"
        print(f"{status} ({x:.1f}, {y:.1f}): {description} - 检测结果={result}, 预期={expected}")
        assert result == expected, f"碰撞检测失败: {description}"

    print("✓ 碰撞检测功能正常")


def test_obstacle_reset():
    """测试障碍物重置功能"""
    print("\n" + "=" * 60)
    print("测试 3: 障碍物重置")
    print("=" * 60)

    obs_manager = ObstacleManager(10.0)

    # 第一次重置
    obs_manager.reset(num_obstacles=5)
    first_obstacles = obs_manager.obstacles.copy()
    print(f"✓ 第一次重置: 生成 {len(first_obstacles)} 个障碍物")

    # 第二次重置
    obs_manager.reset(num_obstacles=8)
    second_obstacles = obs_manager.obstacles.copy()
    print(f"✓ 第二次重置: 生成 {len(second_obstacles)} 个障碍物")

    assert len(first_obstacles) != len(second_obstacles), "重置后障碍物数量应该改变"
    print("✓ 障碍物重置功能正常")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试障碍物管理模块")
    print("=" * 60)

    try:
        test_obstacle_creation()
        test_collision_detection()
        test_obstacle_reset()

        print("\n" + "=" * 60)
        print("✓ 所有障碍物管理测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        sys.exit(1)