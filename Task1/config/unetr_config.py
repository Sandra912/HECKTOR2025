"""UNETR specific configuration."""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class UNETRConfig(BaseConfig):
    """Configuration for UNETR model."""
    
    # Model specific experiment name
    experiment_name: str = "unetr"
    
    # UNETR architecture parameters
    img_size: tuple = (128, 128, 128)  # Must match spatial_size
    # Decoder最终要恢复到的feature_size
    feature_size: int = 16
    # 一个3D patch -> embedding(768维)
    # transformer内部特征维度
    hidden_size: int = 768
    # transformer前馈网络隐藏层的大小
    # transformer每层：Attention -> MLP -> Attention -> MLP
    # 768 -> 3072 -> 768
    mlp_dim: int = 3072
    # 多头注意力的头数（transformer用12种方式关注图像的不同区域）
    # 每个head维度 768/12 = 64,  每个head处理64维的信息
    num_heads: int = 12
    dropout_rate: float = 0.0
    norm_name: str = "instance"
    res_block: bool = True
    conv_block: bool = True
    spatial_dims: int = 3
