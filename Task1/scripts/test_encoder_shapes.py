import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

# 关键：导入你写的 encoder
from models.ssl_encoder import MonaiExactEncoder

model = MonaiExactEncoder(
    spatial_dims=3,
    in_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    dropout=0.0,
    num_res_units=2,
)

x = torch.randn(2, 1, 128, 128, 80)

feats = model(x)

for i, f in enumerate(feats, 1):
    print(f"x{i}: {tuple(f.shape)}")