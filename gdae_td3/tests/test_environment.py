"""
测试完整仿真环境
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math
from gdae_td3.src.environment.simulator import RobotSimulator


def test_environment_initialization():
    """测试环境初始化"""
    print("\n" + "=" * 60)
    print("测试 1: 环境初始化")
    print("=" * 60)

    env = RobotSimulator(map_size=10.0, laser_range=5.0, laser_dim=20)

    assert env.map_size == 10.0, "地图尺寸不匹配"
    assert env.laser_range == 5.0, "激光范围不匹配"
    assert env.laser_dim == 20, "激光维度不匹配"

    print(f"✓ 地图尺寸: {env.map_size}m × {env.map_size}m")
    print(f"✓ 激光范围: {env.laser_range}m")
    print(f"✓ 激光维度: {env.laser_dim}")
    print(f"✓ 速度限制: 线速度={env.max_linear_velocity}m/s, 角速度={env.max_angular_velocity}rad/s")
    print("✓ 环境初始化成功")


def test_environment_reset():
    """测试环境重置"""
    print("\n" + "=" * 60)
    print("测试 2: 环境重置")
    print("=" * 60)

    env = RobotSimulator()

    # 执行多次重置
    for i in range(3):
        obs = env.reset()

        print(f"\n  重置 {i + 1}:")
        print(f"    机器人位置: ({env.x:.2f}, {env.y:.2f}), 朝向: {math.degrees(env.theta):.1f}")
        print(f"    目标位置: ({env.goal_x:.2f}, {env.goal_y:.2f})")
        print(f"    激光数据维度: {len(obs['laser'])}")
        print(f"    机器人状态维度: {len(obs['robot_state'])}")

        # 验证观测格式
        assert 'laser' in obs, "观测缺少 laser 数据"
        assert 'robot_state' in obs, "观测缺少 robot_state 数据"
        assert len(obs['laser']) == 20, "激光数据维度不正确"
        assert len(obs['robot_state']) == 2, "机器人状态维度不正确"

        # 验证机器人和目标位置在地图内
        assert -5.0 <= env.x <= 5.0, "机器人 X 坐标超出地图"
        assert -5.0 <= env.y <= 5.
        0, "机器人 Y 坐标超出地图"
        assert -5.0 <= env.goal_x <= 5.0, "目标 X 坐标超出地图"
        assert -5.0 <= env.goal_y <= 5.0, "目标 Y 坐标超出地图"

    print("\n✓ 环境重置测试通过")


def test_observation_format():
    """测试观测数据格式"""
    print("\n" + "=" * 60)
    print("测试 3: 观测数据格式")
    print("=" * 60)

    env = RobotSimulator()
    obs = env.reset()

    # 检查激光数据
    laser_data = obs['laser']
    print(f"\n  激光数据:")
    print(f"    维度: {len(laser_data)}")
    print(f"    最小值: {min(laser_data):.2f}m")
    print(f"    最大值: {max(laser_data):.2f}m")
    print(f"    平均值: {np.mean(laser_data):.2f}m")

    assert all(0 <= d <= env.laser_range for d in laser_data), "激光数据超出范围"

    # 检查机器人状态
    robot_state = obs['robot_state']
    distance, angle = robot_state

    print(f"\n  机器人状态:")
    print(f"    到目标距离: {distance:.2f}m")
    print(f"    相对角度: {math.degrees(angle):.1f}°")

    assert distance >= 0, "距离不能为负"
    assert -math.pi <= angle <= math.pi, "角度超出范围"

    print("\n✓ 观测数据格式正确")


def test_step_function():
    """测试环境 step 功能"""
    print("\n" + "=" * 60)
    print("测试 4: Step 功能")
    print("=" * 60)

    env = RobotSimulator(
        max_linear_vel=0.5,
        max_angular_vel=2.0,
        time_step=0.1
    )
    obs = env.reset()

    initial_x, initial_y, initial_theta = env.x, env.y, env.theta
    print(f"\n  初始状态:")
    print(f"    位置: ({initial_x:.2f}, {initial_y:.2f})")
    print(f"    朝向: {math.degrees(initial_theta):.1f}")

    # 执行动作：前进
    action = [0.3, 0.0]  # 线速度 0.3, 角速度 0
    next_obs, reward, done, info = env.step(action)

    print(f"\n  执行动作: 线速度={action[0]}, 角速度={action[1]}")
    print(f"    新位置: ({env.x:.2f}, {env.y:.2f})")
    print(f"    新朝向: {math.degrees(env.theta):.1f}°")
    print(f"    奖励: {reward:.2f}")
    print(f"    完成: {done}")
    print(f"    到目标距离: {info['distance_to_goal']:.2f}m")

    # 验证机器人确实移动了
    distance_moved = math.sqrt((env.x - initial_x) ** 2 + (env.y - initial_y) ** 2)
    print(f"    移动距离: {distance_moved:.3f}m")

    assert distance_moved > 0, "机器人应该移动"
    assert 'distance_to_goal' in info, "info 缺少 distance_to_goal"
    assert 'collision' in info, "info 缺少 collision"

    print("\n✓ Step 功能正常")


def test_collision_detection():
    """测试碰撞检测"""
    print("\n" + "=" * 60)
    print("测试 5: 碰撞检测")
    print("=" * 60)

    env = RobotSimulator()
    env.reset()

    # 手动将机器人移动到障碍物上
    if len(env.obstacles.obstacles) > 0:
        obs = env.obstacles.obstacles[0]
        env.x = obs['x']
        env.y = obs['y']

        print(f"  将机器人移动到障碍物位置: ({env.x:.2f}, {env.y:.2f})")

        # 执行一步
        action = [0.0, 0.0]
        next_obs, reward, done, info = env.step(action)

        print(f"    碰撞检测: {info['collision']}")
        print(f"    奖励: {reward:.2f}")
        print(f"    Episode 结束: {done}")

        assert info['collision'] == True, "应该检测到碰撞"
        assert reward == -100.
        0, "碰撞奖励应该是 -100"
        assert done == True, "碰撞后应该结束 Episode"

        print("✓ 碰撞检测正常")
    else:
        print("⚠ 无障碍物，跳过碰撞测试")


def test_goal_reaching():
    """测试目标到达"""
    print("\n" + "=" * 60)
    print("测试 6: 目标到达")
    print("=" * 60)

    env = RobotSimulator()
    env.reset()

    # 手动将机器人移动到目标附近
    env.x = env.goal_x
    env.y = env.goal_y

    print(f"  将机器人移动到目标位置: ({env.x:.2f}, {env.y:.2f})")

    # 执行一步
    action = [0.0, 0.0]
    next_obs, reward, done, info = env.step(action)

    print(f"    到目标距离: {info['distance_to_goal']:.2f}m")
    print(f"    奖励: {reward:.2f}")
    print(f"    Episode 结束: {done}")

    assert info['distance_to_goal'] < env.goal_reach_threshold, "应该到达目标"
    assert reward == 100.0, "到达目标奖励应该是 100"
    assert done == True, "到达目标后应该结束 Episode"

    print("✓ 目标到达检测正常")


def test_reward_function():
    """测试奖励函数"""
    print("\n" + "=" * 60)
    print("测试 7: 奖励函数")
    print("=" * 60)

    env = RobotSimulator()
    env.reset()

    # 测试不同场景的奖励
    test_cases = [
        # (distance, collision, action, 描述)
        (0.2, False, [0.5, 0.0], "接近目标"),
        (5.0, False, [0.3, 0.0], "远离目标"),
        (2.0, False, [0.5, 0.5], "高速转向"),
        (2.0, False, [0.1, 0.0], "缓慢前进"),
    ]

    print("\n  奖励函数测试:")
    for distance, collision, action, desc in test_cases:
        reward = env.compute_reward(distance, collision, action)
        print(f"    {desc}: 距离={distance:.1f}m, 动作={action} → 奖励={reward:.2f}")

    print("\n✓ 奖励函数正常")


def test_multiple_episodes():
    """测试多个 Episode"""
    print("\n" + "=" * 60)
    print("测试 8: 多 Episode 运行")
    print("=" * 60)

    env = RobotSimulator()
    num_episodes = 5
    max_steps = 50

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # 随机动作
            action = [np.random.uniform(0, 0.5), np.random.uniform(-0.5, 0.5)]
            next_obs, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1

            if done:
                break

        print(f"  Episode {ep + 1}: 步数={steps}, 总奖励={total_reward:.2f}, "
              f"结束原因={'碰撞' if info['collision'] else '到达目标' if info['distance_to_goal'] < 0.3 else '超时'}")

    print("\n✓ 多 Episode 运行正常")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试仿真环境")
    print("=" * 60)

    try:
        test_environment_initialization()
        test_environment_reset()
        test_observation_format()
        test_step_function()
        test_collision_detection()
        test_goal_reaching()
        test_reward_function()
        test_multiple_episodes()

        print("\n" + "=" * 60)
        print("✓ 所有环境测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        sys.exit(1)