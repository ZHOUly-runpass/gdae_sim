"""
快速训练脚本
"""
from src.training.train_td3 import TD3Trainer

if __name__ == "__main__":
    # 训练配置（可根据需要调整）
    config = {
        # 训练参数
        'max_timesteps': int(1e6),  # 100万步（可调整为5e6进行完整训练）
        'eval_freq': int(5e3),
        'save_freq': int(5e4),
        'batch_size': 256,

        # 探索参数
        'expl_noise_start': 1.0,
    'expl_noise_end': 0.1,
    'expl_noise_decay_steps': int(5e5),

    # 保存路径
    'save_dir': 'models',
    'log_dir': 'runs',
    'model_name': 'TD3_velodyne',
    }

    # 创建训练器并开始训练
    trainer = TD3Trainer(config)
    trainer.train()