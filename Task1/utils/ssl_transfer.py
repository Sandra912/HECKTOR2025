import torch
from models.ssl_encoder import MonaiExactEncoder


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

    blocks.append(seq[1].submodule)  # bottom

    return blocks

# whitebox.down1  → monai down1
# whitebox.down2  → monai down2
# whitebox.down3  → monai down3
# whitebox.down4  → monai down4
# whitebox.bottom → monai bottom
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

        assert list(src_sd.keys()) == list(dst_sd.keys()), f"Block {i} key mismatch"
        for k in src_sd:
            assert src_sd[k].shape == dst_sd[k].shape, f"Block {i} shape mismatch at {k}"

        dst.load_state_dict(src_sd, strict=True)


def load_ssl_encoder_into_monai_unet(model, config, ckpt_path, map_location="cpu"):
    """
    Load pretrained MonaiExactEncoder weights from checkpoint and copy them
    into the encoder path of current MONAI UNet.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    whitebox = MonaiExactEncoder.from_config(config)
    whitebox.load_state_dict(ckpt["encoder_state_dict"], strict=True)

    copy_encoder_weights(whitebox, model.unet)
    return model