#!/usr/bin/env python3
import os
import random
import numpy as np
import matplotlib.pyplot as plt


NPZ_DIR = "/home/mi2488/hot/datasets/AutoPET/fdg_bodycrop_ssl_npz"
OUT_DIR = "/home/mi2488/hot/projects/HECKTOR2025/Task1/outputs_debug/last40_check"

N_CASES = 8
SEED = 42

# last 40% = image[..., int(Z * 0.60):]
Z_START_FRACTION_LAST40 = 0.60


def normalize_for_show(x):
    x = x.astype(np.float32)

    if np.all(x == x.flat[0]):
        return np.zeros_like(x, dtype=np.float32)

    p1, p99 = np.percentile(x, [1, 99])
    x = np.clip(x, p1, p99)
    x = (x - x.min()) / max(1e-6, x.max() - x.min())
    return x


def make_mip_views(vol_xyz):
    """
    vol_xyz: [X, Y, Z]

    return:
      mip_y: max over Y -> [X, Z]
      mip_x: max over X -> [Y, Z]
      axial_mid: middle axial slice -> [X, Y]
    """
    mip_y = vol_xyz.max(axis=1)  # [X, Z]
    mip_x = vol_xyz.max(axis=0)  # [Y, Z]
    axial_mid = vol_xyz[:, :, vol_xyz.shape[-1] // 2]  # [X, Y]
    return mip_y, mip_x, axial_mid


def show_panel(case_id, img_xyz, out_path):
    z = img_xyz.shape[-1]

    z40 = int(z * 0.40)
    z50 = int(z * 0.50)
    z60 = int(z * 0.60)

    first40 = img_xyz[..., :z40]
    first50 = img_xyz[..., :z50]
    first60 = img_xyz[..., :z60]

    last60 = img_xyz[..., z40:]
    last50 = img_xyz[..., z50:]
    last40 = img_xyz[..., z60:]

    images = [
        ("original full body crop", img_xyz),
        ("first 40%: image[..., :0.40Z]", first40),
        ("last 40%: image[..., 0.60Z:]", last40),
        ("first 60%: image[..., :0.60Z]", first60),
        ("last 60%: image[..., 0.40Z:]", last60),
    ]

    fig, axes = plt.subplots(len(images), 3, figsize=(13, 16))

    for row, (name, vol) in enumerate(images):
        mip_y, mip_x, axial = make_mip_views(vol)

        views = [
            ("MIP over Y  [X,Z]", mip_y),
            ("MIP over X  [Y,Z]", mip_x),
            ("middle axial [X,Y]", axial),
        ]

        for col, (view_name, arr) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(normalize_for_show(arr).T, cmap="gray", origin="lower")
            ax.set_title(f"{name}\n{view_name}\nshape={vol.shape}")
            ax.axis("off")

    fig.suptitle(
        f"{case_id} | Z={z} | "
        f"last40 starts at z={z60}, last60 starts at z={z40}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted([
        f for f in os.listdir(NPZ_DIR)
        if f.endswith(".npz")
    ])

    if len(files) == 0:
        raise RuntimeError(f"No npz files found in {NPZ_DIR}")

    random.seed(SEED)
    chosen = random.sample(files, min(N_CASES, len(files)))

    print(f"NPZ_DIR: {NPZ_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Checking {len(chosen)} cases")
    print("Goal: visually check whether last 40% = upper/head-side 40%")

    for f in chosen:
        path = os.path.join(NPZ_DIR, f)
        data = np.load(path, allow_pickle=True)

        img = data["image"][0]  # [X, Y, Z]
        meta = data["meta"].item() if "meta" in data else {}

        case_id = meta.get("case_id", f.replace(".npz", ""))
        out_path = os.path.join(OUT_DIR, f"{case_id}_last40_check.png")

        z = img.shape[-1]
        z60 = int(z * Z_START_FRACTION_LAST40)

        print("case:", case_id)
        print("  shape:", img.shape)
        print("  Z:", z)
        print("  last40 z_start:", z60)
        print("  last40 shape:", img[..., z60:].shape)
        print("  spacing:", meta.get("spacing"))
        print("  direction:", meta.get("direction"))
        print("  save:", out_path)

        show_panel(case_id, img, out_path)

    print("\nDone.")
    print(f"Open images in: {OUT_DIR}")


if __name__ == "__main__":
    main()