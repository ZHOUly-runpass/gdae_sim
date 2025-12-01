"""
验证环境修复
"""
import sys
import os
sys.path. insert(0, os.path. join(os.path.dirname(__file__), 'src'))

import numpy as np
from environment.simulator import RobotSimulator

print("=" * 70)
print("验证环境修复")
print("=" * 70)

# 创建环境（使用默认参数）
print("\n1. 测试默认参数:")
env_default = RobotSimulator()
print(f"   max_linear_velocity = {env_default.max_linear_velocity} m/s")
print(f"   max_angular_velocity = {env_default.max_angular_velocity} rad/s ({np.degrees(env_default.max_angular_velocity):.1f}°/s)")
print(f"   time_step = {env_default.time_step} s")

# 创建环境（显式指定参数）
print("\n2. 测试显式参数:")
env = RobotSimulator(
    velocity_limits=(0.5, 2.0),
    time_step=0.1
)
print(f"   max_linear_velocity = {env.max_linear_velocity} m/s")
print(f"   max_angular_velocity = {env. max_angular_velocity} rad/s ({np.degrees(env. max_angular_velocity):.1f}°/s)")
print(f"   time_step = {env.time_step} s")

# 检查别名
print("\n3. 检查别名:")
print(f"   max_linear_vel (别名) = {env.max_linear_vel} m/s")
print(f"   max_angular_vel (别名) = {env.max_angular_vel} rad/s")

# 重置环境
obs = env.reset()
initial_x, initial_y, initial_theta = env.x, env.y, env.theta

print(f"\n4. 初始状态:")
print(f"   位置: ({initial_x:.2f}, {initial_y:.2f})")
print(f"   朝向: {np.degrees(initial_theta):.2f}°")
print(f"   目标: ({env.goal_x:.2f}, {env.goal_y:.2f})")
print(f"   距离: {obs['robot_state'][0]:.2f}m")

# 测试动作执行
test_action = [0.5, -0.1]  # linear=0.5, angular=-0. 1
print(f"\n5. 测试动作: linear={test_action[0]}, angular={test_action[1]}")

actual_linear = test_action[0] * env.max_linear_vel
actual_angular = test_action[1] * env.max_angular_vel
print(f"   缩放后实际速度:")
print(f"   - linear_vel = {test_action[0]} × {env.max_linear_vel} = {actual_linear:.3f} m/s")
print(f"   - angular_vel = {test_action[1]} × {env.max_angular_vel} = {actual_angular:.3f} rad/s ({np.degrees(actual_angular):.1f}°/s)")

# 执行5步
print(f"\n6. 执行5步观察变化:")
for step in range(5):
    obs, reward, done, info = env. step(test_action)
    theta_change = np.degrees(env.theta - initial_theta)
    distance_moved = np.sqrt((env.x - initial_x)**2 + (env.y - initial_y)**2)

    print(f"   步骤 {step+1}: θ = {theta_change:+7.2f}°, "
          f"pos = ({env.x:+6.2f}, {env.y:+6.2f}), "
          f"dist = {distance_moved:.3f}m, "
          f"reward = {reward:+6.2f}")

    if done:
        if info['collision']:
            print(f"      → 碰撞！")
        elif info['reach_goal']:
            print(f"      → 到达目标！")
        break

# 理论计算
total_theta_change = np.degrees(env.theta - initial_theta)
expected_theta = np.degrees(actual_angular * env.time_step * min(step + 1, 5))

print(f"\n7. 结果验证:")
print(f"   实际朝向变化: {total_theta_change:+.2f}°")
print(f"   理论朝向变化: {expected_theta:+.2f}°")
print(f"   误差: {abs(total_theta_change - expected_theta):.2f}°")

if abs(total_theta_change - expected_theta) < 1.0:
    print(f"\n✅ 环境修复成功！")
    print(f"   - 速度缩放正常工作")
    print(f"   - 角速度从 {test_action[1]} 缩放到 {actual_angular:.3f} rad/s")
    print(f"   - 转向幅度: {total_theta_change:+.2f}° 符合预期")
else:
    print(f"\n⚠️ 警告：实际值与理论值偏差 {abs(total_theta_change - expected_theta):.2f}°")

print("=" * 70)