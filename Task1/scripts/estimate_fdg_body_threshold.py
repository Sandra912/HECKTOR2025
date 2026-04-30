"""
Cases: 1014
Total sampled voxels: 20280000
min:    0.000000
p0.5:   0.000037
p1:     0.000045
p2:     0.000056
p5:     0.000091
p10:    0.000145
p20:    0.000246
median: 0.000749
max:    432.372589
================================
Recommended BODY_THRESHOLD: 0.010000
"""

#!/usr/bin/env python3
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

RAW_ROOT = "/home/mi2488/hot/datasets/AutoPET/PSMA-FDG-PET-CT-Lesions_v2"
IMAGES_DIR = os.path.join(RAW_ROOT, "imagesTr")

SAMPLE_PER_CASE = 20000
SEED = 42
NUM_WORKERS = 32


def sample_one(args):
    pet_path, seed = args

    rng = np.random.default_rng(seed)

    img = sitk.ReadImage(pet_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    positive = arr[arr > 0]

    if positive.size == 0:
        return None

    n = min(SAMPLE_PER_CASE, positive.size)
    sample = rng.choice(positive, size=n, replace=False)

    return sample.astype(np.float32)


def main():
    pet_files = sorted([
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().startswith("fdg_") and f.endswith("_0001.nii.gz")
    ])

    print(f"Found FDG PET files: {len(pet_files)}")

    worker_count = min(NUM_WORKERS, cpu_count(), len(pet_files))
    print(f"Using {worker_count} workers")

    tasks = [
        (p, SEED + i)
        for i, p in enumerate(pet_files)
    ]

    all_samples = []

    with Pool(processes=worker_count) as pool:
        for sample in tqdm(
            pool.imap_unordered(sample_one, tasks, chunksize=1),
            total=len(tasks),
            desc="Sampling PET intensities",
            unit="case",
            dynamic_ncols=True,
        ):
            if sample is not None:
                all_samples.append(sample)

    all_samples = np.concatenate(all_samples)

    print("================================")
    print(f"Cases: {len(pet_files)}")
    print(f"Total sampled voxels: {len(all_samples)}")
    print(f"min:    {np.min(all_samples):.6f}")
    print(f"p0.5:   {np.percentile(all_samples, 0.5):.6f}")
    print(f"p1:     {np.percentile(all_samples, 1):.6f}")
    print(f"p2:     {np.percentile(all_samples, 2):.6f}")
    print(f"p5:     {np.percentile(all_samples, 5):.6f}")
    print(f"p10:    {np.percentile(all_samples, 10):.6f}")
    print(f"p20:    {np.percentile(all_samples, 20):.6f}")
    print(f"median: {np.median(all_samples):.6f}")
    print(f"max:    {np.max(all_samples):.6f}")

    recommended = max(np.percentile(all_samples, 1), 0.01)
    print("================================")
    print(f"Recommended BODY_THRESHOLD: {recommended:.6f}")


if __name__ == "__main__":
    main()