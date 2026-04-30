#!/usr/bin/env python3
import os
import json
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


RAW_ROOT = "/home/mi2488/hot/datasets/AutoPET/PSMA-FDG-PET-CT-Lesions_v2"

IMAGES_DIR = os.path.join(RAW_ROOT, "imagesTr")

OUT_DIR = "/data/AutoPET/fdg_ssl_npz"
SPLIT_FILE = "/home/mi2488/hot/projects/HECKTOR2025/Task1/config/autopet_fdg_ssl_split_80_20.json"

TARGET_SPACING = (4.0, 4.0, 3.0)
SEED = 42
NUM_WORKERS = 8


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
    resampler.SetSize(out_size)
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
    return arr.astype(np.float16)


def process_one(pet_path):
    case_id = os.path.basename(pet_path).replace("_0001.nii.gz", "")
    out_path = os.path.join(OUT_DIR, f"{case_id}_pet.npz")

    if os.path.exists(out_path):
        return out_path

    img = sitk.ReadImage(pet_path)
    img = reorient_to_lps(img)
    img = resample_image_to_spacing(img, TARGET_SPACING)

    arr = sitk_to_numpy_channel_first(img)

    np.savez_compressed(
        out_path,
        image=arr,
        meta={
            "case_id": case_id,
            "raw_path": pet_path,
            "target_spacing": TARGET_SPACING,
            "size": tuple(int(x) for x in img.GetSize()),
            "spacing": tuple(float(x) for x in img.GetSpacing()),
            "origin": tuple(float(x) for x in img.GetOrigin()),
            "direction": tuple(float(x) for x in img.GetDirection()),
        },
    )

    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pet_files = sorted([
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().startswith("fdg_") and f.endswith("_0001.nii.gz")
    ])

    print(f"Found FDG PET files: {len(pet_files)}")

    if len(pet_files) == 0:
        raise RuntimeError("No FDG PET files found.")

    worker_count = min(NUM_WORKERS, cpu_count())
    print(f"Using {worker_count} workers")

    with Pool(processes=worker_count) as pool:
        npz_paths = list(tqdm(
            pool.imap_unordered(process_one, pet_files, chunksize=1),
            total=len(pet_files),
            desc="Preprocessing FDG PET",
            unit="case",
            dynamic_ncols=True,
        ))

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
    print(f"Saved npz to: {OUT_DIR}")
    print(f"Saved split to: {SPLIT_FILE}")
    print(f"Train: {len(split['train'])}")
    print(f"Val:   {len(split['val'])}")


if __name__ == "__main__":
    main()