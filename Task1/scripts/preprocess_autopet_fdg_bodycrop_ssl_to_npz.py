"""
AutoPET FDG PET body crop pipeline:

1. Load FDG PET image.
2. Reorient to LPS.
3. Resample to target spacing (4.0, 4.0, 3.0) mm.
4. Convert to array [1, X, Y, Z].
5. Build body mask using PET threshold:
   threshold = max(0.01, 1st percentile of positive voxels)
6. Clean mask with 3D opening/closing.
7. Keep the largest connected component as body region.
8. Compute its bounding box [x0, x1, y0, y1, z0, z1].
9. Add padding (8, 8, 8) voxels.
10. Crop PET with padded bbox and save as npz.

Note: cropped image size may differ across cases.
"""

#!/usr/bin/env python3
import os
import json
import random
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import binary_opening, binary_closing, label as cc_label


# =========================
# Paths
# =========================
RAW_ROOT = "/home/mi2488/hot/datasets/AutoPET/PSMA-FDG-PET-CT-Lesions_v2"
IMAGES_DIR = os.path.join(RAW_ROOT, "imagesTr")

OUT_DIR = "/home/mi2488/hot/datasets/AutoPET/fdg_bodycrop_ssl_npz"
SPLIT_FILE = "/home/mi2488/hot/projects/HECKTOR2025/Task1/config/autopet_fdg_bodycrop_ssl_split_80_20.json"


# =========================
# Settings
# =========================
TARGET_SPACING = (4.0, 4.0, 3.0)
SEED = 42

# 你的机器有 96 CPU threads；32 通常比 96 更稳，避免 I/O 爆掉
NUM_WORKERS = 32

# Body crop threshold
BODY_THRESHOLD_MIN = 0.01
BODY_PERCENTILE = 1.0

MIN_CC_VOXELS = 1000
CROP_PADDING = (8, 8, 8)  # X, Y, Z


def reorient_to_lps(img):
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(img)


def compute_new_size(old_size, old_spacing, new_spacing):
    return [
        max(1, int(round(osz * ospc / nspc)))
        for osz, ospc, nspc in zip(old_size, old_spacing, new_spacing)
    ]


def resample_image_to_spacing(image, out_spacing):
    out_size = compute_new_size(image.GetSize(), image.GetSpacing(), out_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize([int(x) for x in out_size])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def sitk_to_numpy_channel_first(img):
    arr = sitk.GetArrayFromImage(img)      # [Z, Y, X]
    arr = np.transpose(arr, (2, 1, 0))     # [X, Y, Z]
    arr = np.expand_dims(arr, axis=0)      # [1, X, Y, Z]
    return arr.astype(np.float32)


def body_crop(arr):
    """
    arr: [1, X, Y, Z]
    return:
      cropped_arr: [1, Xc, Yc, Zc]
      crop_info: dict
    """
    pet = arr[0].astype(np.float32)

    positive = pet[pet > 0]

    if positive.size == 0:
        return arr.astype(np.float16), {
            "body_crop_applied": False,
            "reason": "no_positive_voxels",
            "threshold": None,
            "bbox_xyz": None,
        }

    dynamic_thr = float(np.percentile(positive, BODY_PERCENTILE))
    thr = max(float(BODY_THRESHOLD_MIN), dynamic_thr)

    mask = pet > thr

    structure = np.ones((3, 3, 3), dtype=bool)
    mask = binary_opening(mask, structure=structure)
    mask = binary_closing(mask, structure=structure)

    labeled, num = cc_label(mask)

    if num == 0:
        return arr.astype(np.float16), {
            "body_crop_applied": False,
            "reason": "no_connected_component",
            "threshold": thr,
            "dynamic_threshold": dynamic_thr,
            "bbox_xyz": None,
        }

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    largest_id = int(np.argmax(sizes))
    largest_size = int(sizes[largest_id])

    if largest_size < MIN_CC_VOXELS:
        return arr.astype(np.float16), {
            "body_crop_applied": False,
            "reason": "largest_component_too_small",
            "threshold": thr,
            "dynamic_threshold": dynamic_thr,
            "largest_component_voxels": largest_size,
            "bbox_xyz": None,
        }

    body = labeled == largest_id
    xs, ys, zs = np.where(body)

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    z0, z1 = int(zs.min()), int(zs.max()) + 1

    px, py, pz = CROP_PADDING
    sx, sy, sz = pet.shape

    x0p = max(0, x0 - px)
    x1p = min(sx, x1 + px)
    y0p = max(0, y0 - py)
    y1p = min(sy, y1 + py)
    z0p = max(0, z0 - pz)
    z1p = min(sz, z1 + pz)

    cropped = pet[x0p:x1p, y0p:y1p, z0p:z1p]
    cropped = np.expand_dims(cropped, axis=0).astype(np.float16)

    crop_info = {
        "body_crop_applied": True,
        "threshold": float(thr),
        "dynamic_threshold": float(dynamic_thr),
        "body_threshold_min": float(BODY_THRESHOLD_MIN),
        "body_percentile": float(BODY_PERCENTILE),
        "min_cc_voxels": int(MIN_CC_VOXELS),
        "largest_component_voxels": int(largest_size),
        "crop_padding_xyz": tuple(int(x) for x in CROP_PADDING),
        "bbox_xyz_no_padding": [x0, x1, y0, y1, z0, z1],
        "bbox_xyz": [x0p, x1p, y0p, y1p, z0p, z1p],
        "original_shape": tuple(int(x) for x in arr.shape),
        "cropped_shape": tuple(int(x) for x in cropped.shape),
    }

    return cropped, crop_info


def process_one(pet_path):
    try:
        case_id = os.path.basename(pet_path).replace("_0001.nii.gz", "")
        out_path = os.path.join(OUT_DIR, f"{case_id}_pet.npz")

        if os.path.exists(out_path):
            return ("exists", out_path, case_id, None)

        img = sitk.ReadImage(pet_path)
        img = reorient_to_lps(img)
        img = resample_image_to_spacing(img, TARGET_SPACING)

        arr = sitk_to_numpy_channel_first(img)
        cropped_arr, crop_info = body_crop(arr)

        np.savez_compressed(
            out_path,
            image=cropped_arr,
            meta={
                "case_id": case_id,
                "raw_path": pet_path,
                "target_spacing": TARGET_SPACING,
                "spacing": tuple(float(x) for x in img.GetSpacing()),
                "origin": tuple(float(x) for x in img.GetOrigin()),
                "direction": tuple(float(x) for x in img.GetDirection()),
                "resampled_size_xyz": tuple(int(x) for x in img.GetSize()),
                **crop_info,
            },
        )

        return ("ok", out_path, case_id, crop_info)

    except Exception:
        return ("error", pet_path, None, traceback.format_exc())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)

    pet_files = sorted([
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().startswith("fdg_") and f.endswith("_0001.nii.gz")
    ])

    print(f"Found FDG PET files: {len(pet_files)}")
    print(f"Input dir:  {IMAGES_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Split file: {SPLIT_FILE}")
    print(f"Target spacing: {TARGET_SPACING}")
    print(f"BODY_THRESHOLD_MIN: {BODY_THRESHOLD_MIN}")
    print(f"BODY_PERCENTILE: {BODY_PERCENTILE}")
    print(f"MIN_CC_VOXELS: {MIN_CC_VOXELS}")
    print(f"CROP_PADDING: {CROP_PADDING}")

    if len(pet_files) == 0:
        raise RuntimeError("No FDG PET files found.")

    worker_count = min(NUM_WORKERS, cpu_count(), len(pet_files))
    print(f"Using {worker_count} workers")

    npz_paths = []
    ok_count = 0
    exists_count = 0
    error_count = 0
    fallback_count = 0
    error_cases = []

    with Pool(processes=worker_count) as pool:
        for status, out_path, case_id, info in tqdm(
            pool.imap_unordered(process_one, pet_files, chunksize=1),
            total=len(pet_files),
            desc="Preprocessing FDG PET body crop",
            unit="case",
            dynamic_ncols=True,
        ):
            if status in {"ok", "exists"}:
                npz_paths.append(out_path)
                if status == "ok":
                    ok_count += 1
                    if info is not None and not info.get("body_crop_applied", False):
                        fallback_count += 1
                else:
                    exists_count += 1
            else:
                error_count += 1
                error_cases.append((out_path, info))

    npz_paths = sorted(npz_paths)

    random.seed(SEED)
    random.shuffle(npz_paths)

    n_train = int(0.8 * len(npz_paths))

    split = {
        "train": npz_paths[:n_train],
        "val": npz_paths[n_train:],
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print("================================")
    print(f"Processed new: {ok_count}")
    print(f"Already exists: {exists_count}")
    print(f"Errors: {error_count}")
    print(f"Body crop fallback cases: {fallback_count}")
    print(f"Saved npz to: {OUT_DIR}")
    print(f"Saved split to: {SPLIT_FILE}")
    print(f"Train: {len(split['train'])}")
    print(f"Val:   {len(split['val'])}")

    if error_cases:
        print("\nFirst errors:")
        for p, err in error_cases[:5]:
            print("----")
            print(p)
            print(err)


if __name__ == "__main__":
    main()