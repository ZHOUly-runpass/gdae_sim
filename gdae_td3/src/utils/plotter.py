"""
训练数据绘图工具
用于分析和展示训练过程的各项指标
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


class TrainingPlotter:
    """
    训练数据可视化工具
    支持绘制训练曲线、性能分析、对比图等
    """

    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        初始化绘图工具

        Args:
            style: matplotlib 样式
        """
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                # 如果都失败，使用默认样式
                pass

        # 设置字体
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14

    def plot_training_curves(self, log_dir, save_path=None):
        """
        从 TensorBoard 日志绘制训练曲线

        Args:
            log_dir: TensorBoard 日志目录
            save_path: 保存路径（可选）
        """
        try:
            from tensorboard. backend.event_processing import event_accumulator
        except ImportError:
            print("错误: 需要安装 tensorboard")
            print("运行: pip install tensorboard")
            return

        # 加载事件文件
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    event_files.append(os.path.join(root, file))

        if not event_files:
            print(f"未找到 TensorBoard 日志文件在: {log_dir}")
            return

        print(f"找到 {len(event_files)} 个事件文件")

        # 读取最新的事件文件
        ea = event_accumulator.EventAccumulator(event_files[-1])
        ea.Reload()

        # 获取所有标量标签
        tags = ea.Tags()['scalars']
        print(f"可用标签: {tags}")

        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('TD3 Training Progress', fontsize=16, fontweight='bold')

        # 1. Episode Reward
        if 'train/episode_reward' in tags:
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_scalar(ea, 'train/episode_reward', ax1,
                            'Episode Reward', 'Episodes', 'Reward')

        # 2.  Evaluation Reward
        if 'eval/average_reward' in tags:
            ax2 = fig.add_subplot(gs[0, 2])
            self._plot_scalar(ea, 'eval/average_reward', ax2,
                            'Evaluation Reward', 'Timesteps', 'Avg Reward')

        # 3.  Critic Loss
        if 'train/critic_loss' in tags:
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_scalar(ea, 'train/critic_loss', ax3,
                            'Critic Loss', 'Timesteps', 'Loss', log_scale=True)

        # 4. Actor Loss
        if 'train/actor_loss' in tags:
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_scalar(ea, 'train/actor_loss', ax4,
                            'Actor Loss', 'Timesteps', 'Loss', log_scale=True)

        # 5. Q Values
        if 'train/q1_value' in tags:
            ax5 = fig. add_subplot(gs[1, 2])
            self._plot_scalar(ea, 'train/q1_value', ax5,
                            'Q1 Value', 'Timesteps', 'Q Value')

        # 6.  Success Rate
        if 'eval/success_rate' in tags:
            ax6 = fig. add_subplot(gs[2, 0])
            self._plot_scalar(ea, 'eval/success_rate', ax6,
                            'Success Rate', 'Timesteps', 'Rate', percentage=True)

        # 7. Collision Rate
        if 'eval/collision_rate' in tags:
            ax7 = fig.add_subplot(gs[2, 1])
            self._plot_scalar(ea, 'eval/collision_rate', ax7,
                            'Collision Rate', 'Timesteps', 'Rate', percentage=True)

        # 8. Episode Length
        if 'train/episode_length' in tags:
            ax8 = fig.add_subplot(gs[2, 2])
            self._plot_scalar(ea, 'train/episode_length', ax8,
                            'Episode Length', 'Episodes', 'Steps')

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存至: {save_path}")
        else:
            plt.show()

        plt.close()

    def _plot_scalar(self, ea, tag, ax, title, xlabel, ylabel,
                     log_scale=False, percentage=False, smoothing=0.9):
        """
        绘制单个标量曲线

        Args:
            ea: EventAccumulator 对象
            tag: 标量标签
            ax: matplotlib axis
            title: 标题
            xlabel: x 轴标签
            ylabel: y 轴标签
            log_scale: 是否使用对数刻度
            percentage: 是否显示为百分比
            smoothing: 平滑系数
        """
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        if not values:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontweight='bold')
            return

        # 原始数据
        ax.plot(steps, values, alpha=0.3, color='blue', linewidth=0.5, label='Raw')

        # 平滑数据
        if len(values) > 1:
            smoothed = self._smooth(values, smoothing)
            ax.plot(steps, smoothed, color='red', linewidth=2, label='Smoothed')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if log_scale:
            ax. set_yscale('log')

        if percentage:
            ax. yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    def _smooth(self, values, weight=0.9):
        """指数移动平均平滑"""
        smoothed = []
        last = values[0]
        for value in values:
            smoothed_val = last * weight + (1 - weight) * value
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def plot_comparison(self, results_dict, save_path=None):
        """
        对比多个模型的性能

        Args:
            results_dict: {模型名: 评估结果列表}
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

        # 准备数据
        all_rewards = []
        all_success_rates = []
        all_steps = []
        model_names = []

        for idx, (name, results) in enumerate(results_dict. items()):
            model_names.append(name)

            rewards = [r['reward'] for r in results]
            success = [r['success'] for r in results]
            steps = [r['steps'] for r in results]

            all_rewards.append(rewards)
            all_success_rates.append(sum(success) / len(success))
            all_steps.append(steps)

            # 1. 奖励分布
            axes[0, 0].hist(rewards, alpha=0.5, label=name, bins=20, color=colors[idx])

            # 4. 累计奖励曲线
            cumulative = np.cumsum(rewards)
            axes[1, 1].plot(cumulative, label=name, color=colors[idx], linewidth=2)

        # 2. 成功率柱状图
        x_pos = np.arange(len(model_names))
        axes[0, 1].bar(x_pos, all_success_rates, color=colors[:len(model_names)], alpha=0.7)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1]. set_xticklabels(model_names, rotation=45, ha='right')

        # 3. 步数箱线图
        bp = axes[1, 0]. boxplot(all_steps, labels=model_names, patch_artist=True,
                                widths=0.6)
        for patch, color in zip(bp['boxes'], colors[:len(model_names)]):
            patch.set_facecolor(color)
            patch. set_alpha(0.7)

        # 设置标签
        axes[0, 0].set_title('Reward Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title('Success Rate', fontweight='bold')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        axes[1, 0].set_title('Steps Distribution', fontweight='bold')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        axes[1, 1].set_title('Cumulative Reward', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存至: {save_path}")
        else:
            plt. show()

        plt. close()

    def plot_heatmap(self, trajectory_data, map_size=10.0, save_path=None):
        """
        绘制轨迹热力图

        Args:
            trajectory_data: 轨迹数据列表，支持多种格式：
                            - List[List[tuple]]: [[(x1, y1), (x2, y2), ...], ...]
                            - List[np.ndarray]: [array([[x1, y1], [x2, y2]]), ...]
                            - List[dict]: [{'trajectory': array(...)}, ...]
            map_size: 地图尺寸
            save_path: 保存路径
        """
        # 创建网格
        grid_size = 50
        heatmap = np.zeros((grid_size, grid_size))

        # 处理不同格式的轨迹数据
        processed_trajectories = []

        for idx, traj_item in enumerate(trajectory_data):
            # 情况1: 字典格式 {'trajectory': array(...)}
            if isinstance(traj_item, dict):
                if 'trajectory' in traj_item:
                    trajectory = traj_item['trajectory']
                else:
                    print(f"警告: 轨迹 {idx} 字典中缺少 'trajectory' 键，跳过")
                    continue
            # 情况2: 直接是 numpy 数组或列表
            else:
                trajectory = traj_item

            # 转换为 numpy 数组
            if not isinstance(trajectory, np.ndarray):
                try:
                    trajectory = np. array(trajectory)
                except Exception as e:
                    print(f"警告: 无法转换轨迹 {idx} 数据，错误: {e}，跳过")
                    continue

            # 确保是 2D 数组 (N, 2)
            if trajectory.ndim == 1:
                # 如果是 1D 数组，尝试 reshape
                if len(trajectory) % 2 == 0:
                    trajectory = trajectory. reshape(-1, 2)
                else:
                    print(f"警告: 轨迹 {idx} 数据维度错误（无法 reshape），跳过")
                    continue

            if len(trajectory. shape) != 2 or trajectory.shape[1] != 2:
                print(f"警告: 轨迹 {idx} 数据应该是 (N, 2) 形状，实际是 {trajectory.shape}，跳过")
                continue

            processed_trajectories.append(trajectory)

        if not processed_trajectories:
            print("错误: 没有有效的轨迹数据")
            return

        print(f"成功处理 {len(processed_trajectories)}/{len(trajectory_data)} 条轨迹")

        # 统计每个网格的访问次数
        total_points = 0
        for trajectory in processed_trajectories:
            for point in trajectory:
                try:
                    x, y = float(point[0]), float(point[1])

                    # 转换到网格坐标
                    grid_x = int((x + map_size/2) / map_size * grid_size)
                    grid_y = int((y + map_size/2) / map_size * grid_size)

                    # 边界检查
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        heatmap[grid_y, grid_x] += 1
                        total_points += 1
                except (ValueError, TypeError, IndexError) as e:
                    # 忽略无效的点
                    continue

        # 检查热力图是否为空
        if np.max(heatmap) == 0:
            print("警告: 热力图数据为空，没有有效的轨迹点")
            return

        print(f"热力图统计: 总点数 {total_points}, 最大访问次数 {int(np.max(heatmap))}")

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 9))

        im = ax.imshow(
            heatmap,
            cmap='hot',
            interpolation='bilinear',
            extent=[-map_size/2, map_size/2, -map_size/2, map_size/2],
            origin='lower',
            alpha=0.8
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visit Count', rotation=270, labelpad=20)

        ax.set_title('Robot Trajectory Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3, color='white')

        # 添加统计信息
        info_text = (
            f'Total Points: {total_points}\n'
            f'Max Visits: {int(np.max(heatmap))}\n'
            f'Trajectories: {len(processed_trajectories)}'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props,
            fontfamily='monospace'
        )

        if save_path:
            plt. savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"热力图已保存至: {save_path}")
        else:
            plt.show()

        plt.close()


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("测试 TrainingPlotter")
    print("=" * 60)

    plotter = TrainingPlotter()

    # ===== 测试 1: 对比图 =====
    print("\n[测试 1] 绘制对比图")
    print("-" * 60)

    results_dict = {
        'Model A': [
            {
                'reward': np.random.randn()*10 + 50,
                'success': np. random.rand() > 0.2,
                'steps': np. random.randint(50, 150)
            }
            for _ in range(100)
        ],
        'Model B': [
            {
                'reward': np.random.randn()*10 + 45,
                'success': np.random.rand() > 0.3,
                'steps': np.random.randint(60, 180)
            }
            for _ in range(100)
        ],
        'Model C': [
            {
                'reward': np.random.randn()*8 + 55,
                'success': np.random.rand() > 0.15,
                'steps': np.random.randint(40, 120)
            }
            for _ in range(100)
        ],
    }

    plotter. plot_comparison(results_dict, 'comparison. png')
    print("✓ 对比图已生成: comparison.png")

    # ===== 测试 2: 热力图 =====
    print("\n[测试 2] 绘制热力图")
    print("-" * 60)

    # 生成测试轨迹数据（多种格式）
    trajectory_data = []

    # 格式1: numpy 数组列表
    print("生成 numpy 数组格式轨迹...")
    for i in range(5):
        # 生成随机轨迹
        num_points = np.random.randint(50, 150)
        x = np.cumsum(np.random.randn(num_points) * 0.2)
        y = np.cumsum(np.random.randn(num_points) * 0.2)
        # 限制在地图范围内
        x = np.clip(x, -4, 4)
        y = np.clip(y, -4, 4)
        trajectory = np.column_stack([x, y])
        trajectory_data. append(trajectory)

    # 格式2: 字典格式（模拟测试结果）
    print("生成字典格式轨迹...")
    for i in range(3):
        num_points = np.random.randint(50, 150)
        # 螺旋轨迹
        t = np.linspace(0, 4*np.pi, num_points)
        r = np.linspace(0, 3, num_points)
        x = r * np.cos(t)
        y = r * np.sin(t)
        trajectory = np.column_stack([x, y])
        trajectory_data.append({'trajectory': trajectory})

    # 格式3: 列表格式
    print("生成列表格式轨迹...")
    for i in range(2):
        num_points = np.random.randint(30, 80)
        x = np.linspace(-3, 3, num_points)
        y = np.sin(x) * 2 + np.random.randn(num_points) * 0.1
        trajectory = [[xi, yi] for xi, yi in zip(x, y)]
        trajectory_data.append(trajectory)

    plotter.plot_heatmap(trajectory_data, map_size=10.0, save_path='heatmap.png')
    print("✓ 热力图已生成: heatmap.png")

    # ===== 测试 3: 训练曲线 =====
    print("\n[测试 3] 训练曲线绘制")
    print("-" * 60)
    print("  (需要实际的 TensorBoard 日志文件)")
    print("  使用方法:")
    print("    plotter.plot_training_curves('runs/TD3_velodyne_xxx', 'training. png')")

    # 如果有实际日志，取消注释下面的代码
    # if os.path.exists('runs'):
    #     log_dirs = [d for d in os. listdir('runs') if d.startswith('TD3_velodyne')]
    #     if log_dirs:
    #         plotter.plot_training_curves(f'runs/{log_dirs[0]}', 'training_curves.png')
    #         print("✓ 训练曲线已生成: training_curves.png")

    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - comparison.png   (模型对比图)")
    print("  - heatmap.png      (轨迹热力图)")