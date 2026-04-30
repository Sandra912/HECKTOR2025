import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.ssl_seg_model import Simple3DSegModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Simple3DSegModel(
        in_channels=1,
        num_classes=3,
        feature_channels=(16, 32, 64, 128, 256),
        dropout=0.0,
    ).to(device)

    x = torch.randn(2, 1, 128, 128, 80).to(device)

    # fake labels in [0, 1, 2]
    y = torch.randint(
        low=0,
        high=3,
        size=(2, 128, 128, 80),
        device=device,
        dtype=torch.long,
    )

    logits = model(x)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)

    print("logits shape:", tuple(logits.shape))
    print("target shape:", tuple(y.shape))
    print("loss:", float(loss.item()))

    loss.backward()
    print("backward ok")


if __name__ == "__main__":
    main()