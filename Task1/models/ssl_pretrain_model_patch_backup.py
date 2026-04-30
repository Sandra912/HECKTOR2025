import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssl_encoder import MonaiExactEncoder


class LocalSSLPretrainModel(nn.Module):
    """
    Patch-level / local contrastive SSL model.

    image
    → encoder
    → x5: [B, 256, 8, 8, 5]
    → 1x1x1 projector
    → z_map: [B, 128, 8, 8, 5]
    → sample local patches
    → z_patches: [B, N, 128]

    | `x5`        | `[B,256,8,8,5]` | encoder 学到的最深层 spatial representation       
    | `z_map`     | `[B,128,8,8,5]` | projector 后的 local contrastive embedding map    
    | `z_patches` |     `[B,N,128]` | 从 `z_map` 采样出来用于算 loss 的 patch embeddings 

    """

    def __init__(self, config, proj_dim=128, hidden_dim=512, num_patches=64):
        super().__init__()

        self.encoder = MonaiExactEncoder.from_config(config)
        self.num_patches = num_patches

        feat_dim = config.channels[-1]  # 256

        self.local_projector = nn.Sequential(
            nn.Conv3d(feat_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, proj_dim, kernel_size=1),
        )

    def forward(self, x, patch_indices=None):
        feats = self.encoder(x)
        x5 = feats[-1]  # [B, 256, 8, 8, 5]

        z_map = self.local_projector(x5)  # [B, 128, 8, 8, 5]
        z_map = F.normalize(z_map, dim=1)

        z_patches, patch_indices = self.sample_patches(z_map, patch_indices)

        return z_patches, patch_indices

    def sample_patches(self, z_map, patch_indices=None):
        """
        z_map: [B, C, D, H, W]
        return:
            z_patches: [B, N, C]
            patch_indices: [N]
        """
        B, C, D, H, W = z_map.shape
        L = D * H * W # 所有空间位置数量

        z_flat = z_map.flatten(2)        # [B, C, L]
        z_flat = z_flat.transpose(1, 2)  # 转置 [B, L, C] 每个位置 → 一个 C维向量
        # z_flat[b, i] = 第 b 张图，第 i 个位置的 embedding
        
        if patch_indices is None:
            n = min(self.num_patches, L)
            patch_indices = torch.randperm(L, device=z_map.device)[:n]

        z_patches = z_flat[:, patch_indices, :]  # [B, N, C]， 每张图 → N个patch → 每个patch是128维向量

        return z_patches, patch_indices #[B, N, C] patch特征，哪些位置被选了