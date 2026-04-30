"""
Representation Learning
SSLPretrainModel = encoder + global average pooling + projector

image
→ encoder
→ x5
→ global average pooling
→ projector MLP with BatchNorm
→ embedding z
→ contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssl_encoder import MonaiExactEncoder


class SSLPretrainModel(nn.Module):
    def __init__(self, config, proj_dim=128, hidden_dim=512):
        super().__init__()

        self.encoder = MonaiExactEncoder.from_config(config)

        # image-level global average pooling
        self.pool = nn.AdaptiveAvgPool3d(1)

        # x5 channel dim, normally 256
        feat_dim = config.channels[-1]

        # Standard SimCLR projector
        # h = encoder representation
        # z = contrastive embedding
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        feats = self.encoder(x)       # (x1, x2, x3, x4, x5)
        x5 = feats[-1]                # [B, 256, D, H, W]

        h = self.pool(x5).flatten(1)  # [B, 256]

        z = self.projector(h)         # [B, 128]
        z = F.normalize(z, dim=1)

        return z