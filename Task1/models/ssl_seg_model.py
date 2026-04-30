"""
SSLSegModel = encoder + decoder + seg head 用来输出分割 mask

把 ssl_encoder, 重新接回一个 decoder 和最终分割头，组成一个能训练的 3D segmentation model

image
→ encoder
→ x1,x2,x3,x4,x5
→ decoder(upsampling + skip connection)
→ segmentation head
→ 3 logits
"""

import torch
import torch.nn as nn

from .ssl_encoder import MonaiExactEncoder


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Simple3DSegModel(nn.Module):
    """
    Simple 3D segmentation model built from:
    encoder + decoder + segmentation head

    This is for compatibility testing before SSL pretraining.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        feature_channels=(16, 32, 64, 128, 256),
        dropout=0.0,
    ):
        super().__init__()

        c1, c2, c3, c4, c5 = feature_channels

        # self.encoder = MonaiExactEncoder(
        #     in_channels=in_channels,
        #     feature_channels=feature_channels,
        #     dropout=dropout,
        # )
        self.encoder = MonaiExactEncoder.from_config(config)

        self.dec4 = DecoderBlock3D(c5, c4, c4)
        self.dec3 = DecoderBlock3D(c4, c3, c3)
        self.dec2 = DecoderBlock3D(c3, c2, c2)
        self.dec1 = DecoderBlock3D(c2, c1, c1)

        self.head = nn.Conv3d(c1, num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        y4 = self.dec4(x5, x4)
        y3 = self.dec3(y4, x3)
        y2 = self.dec2(y3, x2)
        y1 = self.dec1(y2, x1)

        logits = self.head(y1)
        return logits

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)