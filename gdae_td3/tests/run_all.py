"""
运行所有测试脚本
"""
import subprocess
import sys


def run_test(test_file, description):
    """运行单个测试文件"""
    print("\n" + "=" * 80)
    print(f"运行测试: {description}")
    print("=" * 80)

    result = subprocess.run([sys.executable, test_file], capture_output=False)

    if result.returncode != 0:
        print(f"\n✗ {description} 失败")
        return False
    else:
        print(f"\n✓ {description} 通过")
        return True


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("TD3 训练环境完整测试套件")
    print("=" * 80)

    tests = [
        ("test_obstacles.py", "障碍物管理模块"),
        ("test_sensors.py", "激光雷达传感器模块"),
        ("test_environment.py", "仿真环境模块"),
        ("test_integration.py", "集成测试"),
    ]

    results = []
    for test_file, description in tests:
        success = run_test(test_file, description)
        results.append((description, success))

    # 输出总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    for description, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {description}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ 所有测试通过！TD3 训练环境搭建成功！ ✓✓✓")
    else:
        print("✗✗✗ 部分测试失败，请检查错误信息 ✗✗✗")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())