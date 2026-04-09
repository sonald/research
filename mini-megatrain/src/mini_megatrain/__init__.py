"""Teaching-oriented MegaTrain reproduction."""

from .config import Config, DataConfig, ModelConfig, TrainingConfig, estimate_parameter_count, load_config
from .engine import StreamingTrainer
from .model import MiniTransformerLM

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "MiniTransformerLM",
    "StreamingTrainer",
    "estimate_parameter_count",
    "load_config",
]
