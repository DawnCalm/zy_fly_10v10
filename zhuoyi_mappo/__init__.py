"""卓翼杯 10v10 的 MAPPO 训练组件。"""

from .config import EnvConfig, TrainConfig
from .env import Kinematic10v10Env
from .model import MAPPOPolicy

__all__ = ["EnvConfig", "TrainConfig", "Kinematic10v10Env", "MAPPOPolicy"]
