import torch
import torch.nn.functional as F


def local_nt_xent_loss(z1, z2, temperature=0.07):
    """
    z1, z2: [B, N, C]  batch size, 每张图采样的 patch 数, embedding 维度

    Positive:
        z1[b, n] 与 z2[b, n]

    Negative:
        batch 中其他 patch embeddings
    """

    B, N, C = z1.shape

    z1 = z1.reshape(B * N, C)
    z2 = z2.reshape(B * N, C)

    z = torch.cat([z1, z2], dim=0)  # [2BN, C]
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature  # 相似度矩阵, [2BN, 2BN]

    batch_size = z.shape[0]
    mask = torch.eye(batch_size, device=z.device).bool() # 去掉自己
    sim = sim.masked_fill(mask, -1e4)
    
    """
    positives = [
    z1_patch0 的正样本索引,
    z1_patch1 的正样本索引,
    z2_patch0 的正样本索引,
    z2_patch1 的正样本索引,
    ]
    """
    positives = torch.cat([
        torch.arange(B * N, 2 * B * N, device=z.device),
        torch.arange(0, B * N, device=z.device),
    ])

    loss = F.cross_entropy(sim, positives)

    return loss