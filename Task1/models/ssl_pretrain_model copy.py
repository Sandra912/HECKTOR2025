"""
Representation Learning
SSLPretrainModel = encoder + pool + projector    

image
→ encoder
→ x5
→ global average pooling
→ projector MLP
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
        self.pool = nn.AdaptiveAvgPool3d(1) # average pooling

        feat_dim = config.channels[-1]  # x5: 256
        # projector = 两层 MLP， 让 encoder 学“通用特征”，让对比学习只在一个“专用空间”里进行
        # image
        # → encoder
        # → h (representation)   ← 你真正想要的特征
        # → projector
        # → z (embedding)        ← 用来做对比学习
        # → contrastive loss
        # self.projector = nn.Sequential(
        #     nn.Linear(feat_dim, hidden_dim), # 256 → 512
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, proj_dim), #  512 → 128
        # )

        # mean + max pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        feat_dim = config.channels[-1] * 2

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        # feats = self.encoder(x) # (x1, x2, x3, x4, x5)
        # x5 = feats[-1] # 提取最深层特征 , [B, 256, 8, 8, 5]
        # z = self.pool(x5).flatten(1) # x5 → pool → [B, 256] 压缩成向量  ← representation
        # z = self.projector(z) # [B, 128]  ← embedding for contrastive loss
        # z = F.normalize(z, dim=1) #L2 normalization, 每个向量长度 = 1, 变成单位球上的向量
        # return z

        feats = self.encoder(x)
        x5 = feats[-1]

        avg = self.avg_pool(x5).flatten(1)
        mx = self.max_pool(x5).flatten(1)

        h = torch.cat([avg, mx], dim=1)

        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z