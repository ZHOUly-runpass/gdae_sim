"""
视频录制工具
将机器人导航过程录制为视频
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import os


class VideoRecorder:
    """
    视频录制器
    将可视化结果保存为 MP4 或 GIF
    """

    def __init__(self, visualizer, fps=30):
        """
        初始化视频录制器

        Args:
            visualizer: TD3Visualizer 实例
            fps: 帧率
        """
        self.visualizer = visualizer
        self.fps = fps
        self.frames = []

    def capture_frame(self):
        """捕获当前帧"""
        # 将当前图形转换为 numpy 数组
        self.visualizer.fig.canvas.draw()
        frame = np.frombuffer(self.visualizer.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.visualizer.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)

    def save_video(self, filename='navigation. mp4', codec='libx264'):
        """
        保存为 MP4 视频

        Args:
            filename: 文件名
            codec: 视频编解码器
        """
        if not self.frames:
            print("没有捕获的帧")
            return

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')

        im = ax.imshow(self.frames[0])

        def update(frame_idx):
            im.set_data(self.frames[frame_idx])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(self.frames),
                             interval=1000 / self.fps, blit=True)

        # 保存视频
        writer = FFMpegWriter(fps=self.fps, codec=codec, bitrate=5000)
        anim.save(filename, writer=writer)

        plt.close(fig)
        print(f"视频已保存至: {filename}")

    def save_gif(self, filename='navigation.gif'):
        """
        保存为 GIF

        Args:
            filename: 文件名
        """
        if not self.frames:
            print("没有捕获的帧")
            return

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')

        im = ax.imshow(self.frames[0])

        def update(frame_idx):
            im.set_data(self.frames[frame_idx])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(self.frames),
                             interval=1000 / self.fps, blit=True)

        # 保存 GIF
        writer = PillowWriter(fps=self.fps)
        anim.save(filename, writer=writer)

        plt.close(fig)
        print(f"GIF 已保存至: {filename}")

    def clear(self):
        """清空帧缓存"""
        self.frames = []