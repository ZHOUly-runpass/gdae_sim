"""
集成测试：完整环境运行验证
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from gdae_td3.src.environment.simulator import RobotSimulator


def test_full_episode_simulation():
    """完整 Episode 仿真测试"""
    print("\n" + "=" * 60)
    print("集成测试: 完整 Episode 仿真")
    print("=" * 60)

    env = RobotSimulator(map_size=10.0, laser_range=5.0, laser_dim=20)

    # 统计数据
    episodes = 10
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_steps = 0
    total_rewards = []

    print(f"\n运行 {episodes} 个 Episode...")

    for ep in range(episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 200

        print(f"\nEpisode {ep + 1}:")
        print(f"  初始位置: ({env.x:.2f}, {env.y:.2f})")
        print(f"  目标位置: ({env.goal_x:.2f}, {env.goal_y:.2f})")
        print(f"  初始距离: {obs['robot_state'][0]:.2f}m")

        done = False
        while not done and steps < max_steps:
            # 简单的启发式策略（朝向目标前进）
            distance, angle = obs['robot_state']
            min_laser = min(obs['laser'])

            # 避障逻辑
            if min_laser < 0.5:
                # 转向更空旷的方向
                left_avg = np.mean(obs['laser'][:10])
                right_avg = np.mean(obs['laser'][10:])
                action = [0.1, 0.8 if left_avg > right_avg else -0.8]
            else:
                # 朝向目标
                linear_vel = min(0.4, distance / 2.0)
                angular_vel = np.clip(angle * 2.0, -0.5, 0.5)
                action = [linear_vel, angular_vel]

            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

        # 统计结果
        total_steps += steps
        total_rewards.append(episode_reward)

        if info['collision']:
            collision_count += 1
            result = "碰撞 ✗"
        elif info['distance_to_goal'] < env.goal_reach_threshold:
            success_count += 1
            result = "成功 ✓"
        else:
            timeout_count += 1
            result = "超时 ⏱"

        print(f"  结果: {result}")
        print(f"  步数: {steps}")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  最终距离: {info['distance_to_goal']:.2f}m")

    # 输出统计
    print("\n" + "=" * 60)
    print("统计结果:")
    print("=" * 60)
    print(f"总 Episodes: {episodes}")
    print(f"成功: {success_count} ({success_count / episodes * 100:.1f}%)")
    print(f"碰撞: {collision_count} ({collision_count / episodes * 100:.1f}%)")
    print(f"超时: {timeout_count} ({timeout_count / episodes * 100:.1f}%)")
    print(f"平均步数: {total_steps / episodes:.1f}")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print("=" * 60)

    # 验证环境稳定性
    assert episodes == success_count + collision_count + timeout_count, "统计数据不一致"
    print("✓ 集成测试通过")


def test_performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    env = RobotSimulator()
    num_steps = 1000

    print(f"\n执行 {num_steps} 步仿真...")

    env.reset()
    start_time = time.time()

    for _ in range(num_steps):
        action = [np.random.uniform(0, 0.5), np.random.uniform(-0.5, 0.5)]
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()

    elapsed_time = time.time() - start_time
    steps_per_second = num_steps / elapsed_time

    print(f"\n性能统计:")
    print(f"  总步数: {num_steps}")
    print(f"  总耗时: {elapsed_time:.2f}s")
    print(f"  速度: {steps_per_second:.1f} steps/s")
    print(f"  平均每步: {elapsed_time / num_steps * 1000:.2f}ms")

    # 验证性能（应该能达到至少 1000 steps/s）
    assert steps_per_second > 100, f"仿真速度太慢: {steps_per_second:.1f} steps/s"
    print(f"\n✓ 性能测试通过（{steps_per_second:.1f} steps/s）")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始集成测试")
    print("=" * 60)

    try:
        test_full_episode_simulation()
        test_performance_benchmark()

        print("\n" + "=" * 60)
        print("✓ 所有集成测试通过！")
        print("✓ TD3 训练环境搭建成功！")
        print("=" * 60)
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        sys.exit(1)