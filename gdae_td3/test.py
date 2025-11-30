"""
快速测试脚本
"""
from src.training.test_td3 import TD3Tester

if __name__ == "__main__":
    # 模型路径
    model_path = "models/TD3_velodyne_best"

    # 创建测试器
    tester = TD3Tester(model_path, render=True)

    # 测试10个episodes
    stats = tester.test_multiple_episodes(num_episodes=10)

    # 绘制轨迹
    tester.plot_trajectories(stats['results'])