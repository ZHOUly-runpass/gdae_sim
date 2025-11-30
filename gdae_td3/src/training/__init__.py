"""
训练模块
"""
from .train_td3 import TD3Trainer
from .test_td3 import TD3Tester

__all__ = ['TD3Trainer', 'TD3Tester']