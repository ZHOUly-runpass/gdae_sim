# verify_installation.py
import sys


def verify_installation():
    print("=" * 60)
    print("GDAE 仿真环境安装验证")
    print("=" * 60)

    # 检查 Python 版本
    print(f"\n✓ Python 版本: {sys.version.split()[0]}")
    assert sys.version_info >= (3, 8), "Python 版本过低"

    # 检查核心库
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
        assert np.__version__ >= "1.21", "NumPy 版本过低"
    except ImportError:
        print("✗ NumPy 未安装")
        return False

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA 版本: {torch.version.cuda}")
            print(f"  - GPU 数量: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ PyTorch 未安装")
        return False

    try:
        import gym
        print(f"✓ Gym: {gym.__version__}")
    except ImportError:
        print("✗ Gym 未安装")

    try:
        import gymnasium
        print(f"✓ Gymnasium: {gymnasium.__version__}")
    except ImportError:
        print("⚠ Gymnasium 未安装（可选）")

    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib 未安装")
        return False

    try:
        import scipy
        print(f"✓ SciPy: {scipy.__version__}")
    except ImportError:
        print("✗ SciPy 未安装")
        return False

    try:
        import pandas
        print(f"✓ Pandas: {pandas.__version__}")
    except ImportError:
        print("⚠ Pandas 未安装（可选）")

    try:
        from tensorboard import __version__ as tb_version
        print(f"✓ TensorBoard: {tb_version}")
    except ImportError:
        print("⚠ TensorBoard 未安装（推荐）")

    try:
        import yaml
        print(f"✓ PyYAML: {yaml.__version__}")
    except ImportError:
        print("⚠ PyYAML 未安装（推荐）")

    try:
        import tqdm
        print(f"✓ tqdm: {tqdm.__version__}")
    except ImportError:
        print("⚠ tqdm 未安装（推荐）")

    # 测试 PyTorch 基本功能
    print("\n" + "=" * 60)
    print("PyTorch 功能测试")
    print("=" * 60)

    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("✓ 矩阵乘法测试通过")

        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print("✓ GPU 计算测试通过")
    except Exception as e:
        print(f"✗ PyTorch 功能测试失败: {e}")
        return False

    # 测试神经网络
    print("\n" + "=" * 60)
    print("神经网络测试")
    print("=" * 60)

    try:
        import torch.nn as nn

        class TestNet(nn.Module):
            def __init__(self):
                super(TestNet, self).__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 2)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        net = TestNet()
        test_input = torch.randn(5, 10)
        output = net(test_input)
        assert output.shape == (5, 2), "网络输出形状错误"
        print("✓ 神经网络测试通过")
    except Exception as e:
        print(f"✗ 神经网络测试失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ 环境验证完成！所有核心组件工作正常。")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)