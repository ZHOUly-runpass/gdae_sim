"""
TD3 训练脚本
严格按照原项目训练流程实现
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '. .')))

import numpy as np
import torch
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gdae_td3.src.environment.simulator import RobotSimulator
from gdae_td3.src.td3.agent import TD3Agent
from gdae_td3.src.td3.replay_buffer import ReplayBuffer


class TD3Trainer:
    """TD3 训练器"""

    def __init__(self, config=None):
        """初始化训练器"""
        # 默认配置
        self.config = {
            'map_size': 10.0,
            'laser_range': 5.0,
            'laser_dim': 20,
            'state_dim': 24,
            'action_dim': 2,
            'max_action': 1.0,
            'max_timesteps': int(5e6),
            'start_timesteps': int(1e4),
            'batch_size': 256,
            'eval_freq': int(2e3),  #调整从 5e3 降低，更早发现问题
            'save_freq':  int(5e4),
            'discount':  0.99,
            'tau': 0.005,
            'policy_noise':  0.2,
            'noise_clip': 0.5,
            'policy_freq': 2,
            # 'expl_noise_start': 1.0,
            'expl_noise_start': 0.3,
            # 'expl_noise_end': 0.1,
            # 添加梯度裁剪
            'max_grad_norm': 1.0,
            'expl_noise_end': 0.05,
            'expl_noise_decay_steps': int(5e5),
            'random_near_obstacle': True,
            'obstacle_threshold': 0.6,
            'random_action_prob': 0.85,
            'random_action_steps': (8, 15),
            'buffer_size': int(1e6),
            'max_episode_steps': 500,
            'save_dir':  'models',
            'log_dir': 'runs',
            'model_name': 'TD3_velodyne',
            'seed': 0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        if config:
            self.config.update(config)

        self._set_seed(self.config['seed'])

        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self. config['log_dir'], exist_ok=True)

        print("创建训练环境...")
        self.env = RobotSimulator(
            map_size=self.config['map_size'],
            laser_range=self.config['laser_range'],
            laser_dim=self.config['laser_dim'],
            velocity_limits=(0.5, 2.0),
            time_step=0.1
        )

        print("创建 TD3 智能体...")
        self.agent = TD3Agent(
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            max_action=self.config['max_action'],
            device=torch.device(self.config['device']),
            gamma=self.config['discount'],
            tau=self.config['tau'],
            policy_noise=self.config['policy_noise'],
            noise_clip=self.config['noise_clip'],
            policy_freq=self.config['policy_freq']
        )

        print("创建经验回放缓冲区...")
        self.replay_buffer = ReplayBuffer(
            max_size=self.config['buffer_size'],
            seed=self.config['seed']
        )

        log_dir = os.path.join(
            self.config['log_dir'],
            f"{self.config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard 日志目录: {log_dir}")

        self.episode_num = 0
        self.episode_reward = 0
        self.episode_timesteps = 0
        self. total_timesteps = 0
        self.expl_noise = self.config['expl_noise_start']
        self.random_action_count = 0
        self.random_action = None

        print("初始化完成！")

    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _get_state(self, obs):
        """
        构建完整状态向量 - 与参考项目一致

        状态构成:  [laser(20) + distance(1) + theta(1) + action(2)] = 24维
        """
        # 激光数据：直接使用，不归一化
        laser_data = np.array(obs['laser'])

        # 机器人状态
        robot_state = np.array(obs['robot_state'])

        # 当前动作
        action = np.array(obs['action'])

        # 拼接状态
        state = np.concatenate([laser_data, robot_state, action])

        return state

    def _select_action(self, state):
        """选择动作（带探索噪声）"""
        if self.total_timesteps < self.config['start_timesteps']:
            action = np.random.uniform(-1, 1, self.config['action_dim'])
        else:
            action = self.agent.get_action(state)
            noise = np.random.normal(0, self.expl_noise, size=self.config['action_dim'])
            action = (action + noise).clip(-self.config['max_action'], self.config['max_action'])

        # 随机障碍物附近探索策略
        if self.config['random_near_obstacle']:
            # 注意：现在 laser 数据没有归一化，阈值需要调整
            laser_data = state[: 20]
            min_laser = min(laser_data)

            if (np.random.rand() > self.config['random_action_prob'] and
                    min_laser < self.config['obstacle_threshold'] and
                    self.random_action_count < 1):
                self.random_action_count = np.random.randint(*self.config['random_action_steps'])
                self.random_action = np.random.uniform(-1, 1, 2)
                self.random_action[0] = -1

            if self.random_action_count > 0:
                self.random_action_count -= 1
                action = self.random_action. copy()

        return action

    def _update_exploration_noise(self):
        """更新探索噪声"""
        if self. expl_noise > self.config['expl_noise_end']:
            decay = (self.config['expl_noise_start'] - self.config['expl_noise_end']) / \
                    self.config['expl_noise_decay_steps']
            self.expl_noise = max(self.expl_noise - decay, self.config['expl_noise_end'])

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 80)
        print("开始训练 TD3")
        print("=" * 80)
        print(f"最大训练步数: {self.config['max_timesteps']: ,}")
        print(f"评估频率: 每 {self.config['eval_freq']:,} 步")
        print(f"保存频率: 每 {self.config['save_freq']:,} 步")
        print(f"设备: {self.config['device']}")
        print("=" * 80 + "\n")

        # 初始化
        obs = self.env.reset()
        state = self._get_state(obs)  # 不再需要 last_action 参数

        done = False
        self.episode_reward = 0
        self.episode_timesteps = 0

        evaluations = []
        best_reward = -np.inf

        pbar = tqdm(total=self.config['max_timesteps'], desc="Training")

        while self.total_timesteps < self.config['max_timesteps']:

            if done or self.episode_timesteps >= self.config['max_episode_steps']:
                if self.total_timesteps >= self.config['start_timesteps']:
                    train_stats = self.agent.train(
                        self.replay_buffer,
                        batch_size=self.config['batch_size'],
                        discount=self.config['discount'],
                        tau=self.config['tau'],
                        policy_noise=self.config['policy_noise'],
                        noise_clip=self.config['noise_clip'],
                        policy_freq=self.config['policy_freq']
                    )

                    self.writer.add_scalar('train/critic_loss', train_stats['critic_loss'], self.total_timesteps)
                    if train_stats['actor_loss'] is not None:
                        self.writer.add_scalar('train/actor_loss', train_stats['actor_loss'], self.total_timesteps)
                    self.writer.add_scalar('train/q1_value', train_stats['q1_value'], self.total_timesteps)

                self.writer.add_scalar('train/episode_reward', self.episode_reward, self.episode_num)
                self.writer.add_scalar('train/episode_length', self.episode_timesteps, self.episode_num)
                self.writer.add_scalar('train/exploration_noise', self.expl_noise, self.total_timesteps)

                obs = self.env.reset()
                state = self._get_state(obs)

                done = False
                self.episode_reward = 0
                self. episode_timesteps = 0
                self.episode_num += 1

            action = self._select_action(state)

            # 转换动作到环境输入范围
            action_in = [(action[0] + 1) / 2, action[1]]

            next_obs, reward, done, info = self.env.step(action_in)
            next_state = self._get_state(next_obs)

            done_bool = float(done) if self.episode_timesteps < self.config['max_episode_steps'] else 0
            self.replay_buffer.add(state, action, reward, done_bool, next_state)

            state = next_state
            self.episode_reward += reward
            self.episode_timesteps += 1
            self.total_timesteps += 1

            self._update_exploration_noise()

            pbar.update(1)
            pbar.set_postfix({
                'Episode':  self.episode_num,
                'Reward': f'{self.episode_reward:.1f}',
                'Noise': f'{self.expl_noise:.3f}',
                'Buffer':  f'{len(self.replay_buffer):,}'
            })

            if self.total_timesteps % self. config['eval_freq'] == 0:
                eval_reward = self.evaluate(num_episodes=10)
                evaluations.append(eval_reward)

                self.writer.add_scalar('eval/average_reward', eval_reward, self.total_timesteps)

                print(f"\n[评估] Timestep:  {self.total_timesteps:,}, Average Reward: {eval_reward:.2f}")

                if eval_reward > best_reward:
                    best_reward = eval_reward
                    save_path = os.path.join(self.config['save_dir'], f"{self.config['model_name']}_best")
                    self.agent.save(save_path)
                    print(f"保存最佳模型:  {save_path}. pth (Reward: {eval_reward:.2f})")

            if self.total_timesteps % self.config['save_freq'] == 0:
                save_path = os.path.join(
                    self.config['save_dir'],
                    f"{self.config['model_name']}_step_{self.total_timesteps}"
                )
                self.agent.save(save_path)
                print(f"\n保存检查点: {save_path}. pth")

        pbar.close()

        final_path = os.path.join(self.config['save_dir'], f"{self.config['model_name']}_final")
        self.agent.save(final_path)
        print(f"\n训练完成！最终模型保存至: {final_path}.pth")

        self.writer.close()

        return evaluations

    def evaluate(self, num_episodes=10):
        """评估当前策略"""
        avg_reward = 0.0
        success_count = 0
        collision_count = 0

        for _ in range(num_episodes):
            obs = self.env. reset()
            state = self._get_state(obs)

            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < self.config['max_episode_steps']:
                action = self. agent.get_action(state, add_noise=False)
                action_in = [(action[0] + 1) / 2, action[1]]

                next_obs, reward, done, info = self.env.step(action_in)
                next_state = self._get_state(next_obs)

                state = next_state
                episode_reward += reward
                steps += 1

            avg_reward += episode_reward

            if info. get('distance_to_goal', 1.0) < 0.3:
                success_count += 1
            if info.get('collision', False):
                collision_count += 1

        avg_reward /= num_episodes
        success_rate = success_count / num_episodes
        collision_rate = collision_count / num_episodes

        self.writer.add_scalar('eval/success_rate', success_rate, self.total_timesteps)
        self.writer.add_scalar('eval/collision_rate', collision_rate, self.total_timesteps)

        return avg_reward


def main():
    """主函数"""
    config = {
        'max_timesteps': int(5e6),
        'eval_freq': int(5e3),
        'save_freq': int(5e4),
        'batch_size': 256,
        'expl_noise_start': 1.0,
        'expl_noise_end': 0.1,
        'expl_noise_decay_steps': int(5e5),
    }

    trainer = TD3Trainer(config)
    evaluations = trainer.train()

    results_path = os.path.join(trainer.config['save_dir'], 'evaluations.npy')
    np.save(results_path, evaluations)
    print(f"评估结果保存至: {results_path}")


if __name__ == "__main__":
    main()