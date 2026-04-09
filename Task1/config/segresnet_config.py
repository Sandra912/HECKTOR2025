"""SegResNet specific configuration."""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class SegResNetConfig(BaseConfig):
    """Configuration for SegResNet model."""
    
    # Model specific experiment name
    experiment_name: str = "segresnet"
    
    # SegResNet architecture parameters``
    # 下采样每一层有几个残差块（encoder）
    blocks_down: tuple = (1, 2, 2, 4)
    # decoder
    blocks_up: tuple = (1, 1, 1)
    # 第一层的通道数 2 -> 16 (从CT,PET -> 提取16种特征)
    init_filters: int = 16
    in_channels: int = 2  # CT + PET (same as input_channels in BaseConfig)
    out_channels: int = 3  # background + primary tumor + metastatic tumor (same as num_classes in BaseConfig)
    dropout_prob: float = 0.2



