"""
对比最佳模型和最终模型的性能
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from gdae_td3.src.training.test_td3 import TD3Tester
from gdae_td3.src.utils.plotter import TrainingPlotter


def main():
    print("=" * 80)
    print("模型性能对比")
    print("=" * 80)

    # 定义要对比的模型
    models = {
        'Best Model': 'models/TD3_velodyne_best',
        'Final Model': 'models/TD3_velodyne_final',
    }

    results_dict = {}

    # 测试每个模型
    for name, path in models.items():
        print(f"\n{'=' * 60}")
        print(f"测试 {name}: {path}")
        print('=' * 60)

        # 检查模型是否存在
        if not os.path.exists(f"{path}.pth"):
            print(f"⚠ 模型不存在，跳过")
            continue

        # 测试模型
        tester = TD3Tester(path, render=False)
        stats = tester.test_multiple_episodes(num_episodes=50)
        results_dict[name] = stats['results']

        # 打印统计
        print(f"\n结果统计:")
        print(f"  成功率: {stats['success_rate'] * 100:.1f}%")
        print(f"  碰撞率: {stats['collision_rate'] * 100:.1f}%")
        print(f"  平均步数: {stats['avg_steps']:. 1f}")
        print(f"  平均奖励: {stats['avg_reward']:. 2f} ± {stats['std_reward']:.2f}")

    if not results_dict:
        print("\n⚠ 没有可对比的模型")
        return

    # 绘制对比图
    print("\n" + "=" * 80)
    print("生成对比图...")
    print("=" * 80)

    plotter = TrainingPlotter()

    # 1. 性能对比图
    plotter.plot_comparison(results_dict, 'model_comparison. png')
    print("✓ 生成对比图: model_comparison.png")

    # 2. 每个模型的热力图
    for name, results in results_dict.items():
        trajectory_data = [r['trajectory'] for r in results]
        filename = f"heatmap_{name.replace(' ', '_').lower()}.png"
        plotter.plot_heatmap(trajectory_data, map_size=10.0, save_path=filename)
        print(f"✓ 生成热力图: {filename}")

    print("\n" + "=" * 80)
    print("对比完成！")
    print("=" * 80)
    print("\n生成的文件：")
    print("  - model_comparison. png       (性能对比)")
    for name in results_dict.keys():
        filename = f"heatmap_{name.replace(' ', '_').lower()}.png"
        print(f"  - {filename:30} (轨迹热力图)")
    print("=" * 80)


if __name__ == "__main__":
    main()