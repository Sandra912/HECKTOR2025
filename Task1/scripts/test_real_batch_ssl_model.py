# 用项目里真实的一批 PET+label 数据，验证新写的 ssl_encoder + decoder 模型，是否真的能无缝接入当前 HECKTOR 训练数据流并参与训练。
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import UNet3DConfig
from data import get_dataloaders
from models.ssl_seg_model import Simple3DSegModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = UNet3DConfig()
    train_loader, val_loader = get_dataloaders(config, fold=0)

    batch = next(iter(train_loader))

    # 这里根据你 dataloader 的实际 key 可能是 image / label
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    print("images.shape:", tuple(images.shape))
    print("labels.shape:", tuple(labels.shape))
    print("images.dtype:", images.dtype)
    print("labels.dtype:", labels.dtype)

    model = Simple3DSegModel(
        in_channels=config.input_channels,
        num_classes=config.num_classes,
        feature_channels=config.channels,
        dropout=config.dropout,
    ).to(device)

    logits = model(images)

    print("logits.shape:", tuple(logits.shape))

    # 如果 label 是 [B,1,H,W,D]，要 squeeze 掉 channel 维
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]

    labels = labels.long()

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    print("loss:", float(loss.item()))

    loss.backward()
    print("real batch backward ok")


if __name__ == "__main__":
    main()