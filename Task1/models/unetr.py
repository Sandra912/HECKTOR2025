"""UNETR model implementation using MONAI."""

import torch
import torch.nn as nn
from monai.networks.nets import UNETR

from .base_model import BaseModel


class UNETRModel(BaseModel):
    """UNETR model for segmentation using MONAI."""
    
    def __init__(self, config):
        """
        Initialize UNETR model.
        
        Args:
            config: UNETRConfig object
        """
        super().__init__(config)
        
        # Initialize MONAI UNETR
        self.unetr = UNETR(
            in_channels=config.input_channels,
            out_channels=config.num_classes,
            img_size=config.img_size,
            feature_size=config.feature_size, # 解码器的基础特征通道数
            hidden_size=config.hidden_size,
            mlp_dim=config.mlp_dim,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            norm_name=config.norm_name,
            res_block=config.res_block,
            conv_block=config.conv_block,
            spatial_dims=config.spatial_dims
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNETR.
        
        Args:
            x: Input tensor of shape (B, C, H, W, D)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W, D)
        """
        return self.unetr(x)
    
    """
    如果是 Conv → 用 Kaiming 初始化
    如果是 BN → weight=1, bias=0
    如果是 Linear → Normal(0, 0.01)
    """
    def _initialize_weights(self):
        """Initialize model weights."""
        """ Transformer里大量用 Linear: patch embedding, attention (Q, K, V), MLP"""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                # mode="fan_out": 控制输出方差稳定
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # BatchNorm 批归一化= 把每一层的输出“拉回正常范围”，让训练更稳定、更快
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01) #均值为0， 方差0.01
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_model_info(self) -> str:
        """Get model architecture information."""
        params = self.get_parameters()
        
        info = f"""
UNETR Model Information:
------------------------
Architecture: UNETR (MONAI)
Input Channels: {self.config.input_channels}
Output Channels: {self.config.num_classes}
Image Size: {self.config.img_size}
Feature Size: {self.config.feature_size}
Hidden Size: {self.config.hidden_size}
MLP Dim: {self.config.mlp_dim}
Number of Heads: {self.config.num_heads}
Dropout Rate: {self.config.dropout_rate}

Parameters:
-----------
Total Parameters: {params['total_parameters']:,}
Trainable Parameters: {params['trainable_parameters']:,}
Model Size: {params['model_size_mb']:.2f} MB
        """
        
        return info.strip()
