"""UNet3D specific configuration."""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class UNet3DConfig(BaseConfig):
    """Configuration for UNet3D model."""
    
    # Model specific experiment name
    experiment_name: str = "unet3d"
    
    # UNet3D architecture parameters
    spatial_dims: int = 3 #3 dimension:[B, C, D, H, W] (B:batch size, C:channel(CT/PET))
    # 每一层的特征通道数，channels 越大，model越强
    channels: tuple = (16, 32, 64, 128, 256) 
    # 每一层下采样的倍数， 每一层空间尺寸缩小一半（/2)
    strides: tuple = (2, 2, 2, 2)
    # 卷积核大小 3*3*3
    kernel_size: int = 3
    # 上采样的卷积核
    up_kernel_size: int = 3
    # dropout用来防止过拟合（随机断开一部分神经元的连接）
    dropout: float = 0.0
    # 残差单元数量
    # 残差解决的问题：网络很深，可能梯度消失，训练变差
    # 残差：新层只负责微调，原始信息一直被保留（不强迫每一层都学完整变换，只在原来基础上微调）
    # 残差块： 输入x -> Conv -> Conv -> + x -> 输出
    num_res_units: int = 2
    
