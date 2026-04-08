"""SwinUNETR specific configuration."""
# 局部窗口版Transformer(Swin) + Unet Decoder
from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class SwinUNETRConfig(BaseConfig):
    """Configuration for SwinUNETR model."""
    
    # Model specific experiment name
    experiment_name: str = "swinunetr"
    
    # 图像分成多个window,  在每个window内部做attention, shift window, 多层叠加逐渐获得全局信息
    # SwinUNETR architecture parameters
    img_size: tuple = (128, 128, 128)  # Must match spatial_size
    in_channels: int = 2  # CT + PET (same as input_channels in BaseConfig)
    out_channels: int = 3  # background + primary tumor + metastatic tumor (same as num_classes in BaseConfig)
    # Decoder最终要恢复到的feature_size
    feature_size: int = 48
    # 每一层有多少个Swin Transformer block
    depths: tuple = (2, 2, 2, 2)
    # 每一层的注意力头数
    num_heads: tuple = (3, 6, 12, 24)
    drop_rate: float = 0.0
    # attention里的droprate
    attn_drop_rate: float = 0.0
    # 随机丢掉整个层
    dropout_path_rate: float = 0.0
    normalize: bool = True
    use_checkpoint: bool = False
    spatial_dims: int = 3
