import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit
from monai.networks.layers.factories import Act, Norm


class MonaiExactEncoder(nn.Module):
    """
    White-box encoder that matches MONAI UNet down path exactly
    for the current config:
      channels=(16, 32, 64, 128, 256)
      strides=(2, 2, 2, 2)
      num_res_units=2
    """

    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        dropout=0.0,
        num_res_units=2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
    ):
        super().__init__()

        c1, c2, c3, c4, c5 = channels
        s1, s2, s3, s4 = strides

        common = dict(
            spatial_dims=spatial_dims,
            kernel_size=kernel_size,
            subunits=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        
        # 每一层 Conv → Norm → Act → Conv → Norm → + skip
        # ResidualUnit = 带 skip connection 的卷积块
        """
        common 就是一个参数字典(dict), **common  把这些参数全部展开传进去
        common = dict(
            spatial_dims=spatial_dims,
            kernel_size=kernel_size,
            subunits=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        """
        self.down1 = ResidualUnit(
            in_channels=in_channels,
            out_channels=c1,
            strides=s1,
            **common,
        )
        self.down2 = ResidualUnit(
            in_channels=c1,
            out_channels=c2,
            strides=s2,
            **common,
        )
        self.down3 = ResidualUnit(
            in_channels=c2,
            out_channels=c3,
            strides=s3,
            **common,
        )
        self.down4 = ResidualUnit(
            in_channels=c3,
            out_channels=c4,
            strides=s4,
            **common,
        )

        # bottom layer in MONAI UNet is _get_down_layer(..., stride=1)
        self.bottom = ResidualUnit(
            in_channels=c4,
            out_channels=c5,
            strides=1,
            **common,
        )
    
    @classmethod
    def from_config(cls, config):
        return cls(
            spatial_dims=config.spatial_dims,
            in_channels=config.input_channels,
            channels=config.channels,
            strides=config.strides,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            num_res_units=config.num_res_units,
        )
        
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.bottom(x4)
        return x1, x2, x3, x4, x5