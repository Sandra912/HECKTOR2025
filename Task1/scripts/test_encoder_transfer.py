import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import UNet3DConfig
from models.ssl_encoder import MonaiExactEncoder
from models.unet3d import UNet3DModel


def get_monai_down_blocks(monai_unet):
    """
    Extract the 5 encoder blocks from MONAI UNet recursion:
      down1, down2, down3, down4, bottom
    """
    blocks = []

    seq = monai_unet.model
    blocks.append(seq[0])  # down1

    seq = seq[1].submodule
    blocks.append(seq[0])  # down2

    seq = seq[1].submodule
    blocks.append(seq[0])  # down3

    seq = seq[1].submodule
    blocks.append(seq[0])  # down4

    # bottom layer
    blocks.append(seq[1].submodule)

    return blocks


def copy_encoder_weights(whitebox_encoder, monai_unet):
    src_blocks = [
        whitebox_encoder.down1,
        whitebox_encoder.down2,
        whitebox_encoder.down3,
        whitebox_encoder.down4,
        whitebox_encoder.bottom,
    ]
    dst_blocks = get_monai_down_blocks(monai_unet)

    assert len(src_blocks) == len(dst_blocks) == 5

    for i, (src, dst) in enumerate(zip(src_blocks, dst_blocks), 1):
        src_sd = src.state_dict()
        dst_sd = dst.state_dict()

        # 先检查 key 和 shape 是否完全一致
        assert list(src_sd.keys()) == list(dst_sd.keys()), f"Block {i} key mismatch"
        for k in src_sd:
            assert src_sd[k].shape == dst_sd[k].shape, f"Block {i} shape mismatch at {k}"

        dst.load_state_dict(src_sd, strict=True)
        print(f"[OK] copied block {i}")


def register_block_hooks(blocks, outputs_dict):
    hooks = []

    for idx, block in enumerate(blocks, 1):
        name = f"x{idx}"

        def save_output(name):
            def hook(module, inp, out):
                outputs_dict[name] = out.detach().cpu()
            return hook

        hooks.append(block.register_forward_hook(save_output(name)))

    return hooks


def main():
    torch.manual_seed(0)

    config = UNet3DConfig()

    # 1) build white-box encoder
    whitebox = MonaiExactEncoder(
        spatial_dims=config.spatial_dims,
        in_channels=config.input_channels,
        channels=config.channels,
        strides=config.strides,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        num_res_units=config.num_res_units,
    ).eval()

    # 2) build current MONAI segmentation model
    seg_model = UNet3DModel(config).eval()

    # 3) copy white-box encoder weights into MONAI UNet encoder
    copy_encoder_weights(whitebox, seg_model.unet)

    # 4) compare intermediate outputs on same input
    x = torch.randn(1, 1, 128, 128, 80)

    with torch.no_grad():
        white_feats = whitebox(x)

    monai_outputs = {}
    monai_blocks = get_monai_down_blocks(seg_model.unet)
    hooks = register_block_hooks(monai_blocks, monai_outputs)

    with torch.no_grad():
        _ = seg_model.unet(x)

    for h in hooks:
        h.remove()

    print("\n=== compare feature maps ===")
    for i, wf in enumerate(white_feats, 1):
        mf = monai_outputs[f"x{i}"]
        diff = (wf.detach().cpu() - mf).abs().max().item()
        same_shape = tuple(wf.shape) == tuple(mf.shape)
        print(f"x{i}: shape_equal={same_shape}, max_abs_diff={diff:.10f}")

    print("\nDone.")


if __name__ == "__main__":
    main()