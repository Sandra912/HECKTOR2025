import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

DATA_ROOT = "/data/HECKTOR2025/Task1"

def analyze_pet(pet_path):
    img = sitk.ReadImage(pet_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    arr_flat = arr.flatten()

    stats = {
        "min": float(arr_flat.min()),
        "max": float(arr_flat.max()),
        "mean": float(arr_flat.mean()),
        "std": float(arr_flat.std()),
        "p50": float(np.percentile(arr_flat, 50)),
        "p90": float(np.percentile(arr_flat, 90)),
        "p95": float(np.percentile(arr_flat, 95)),
        "p99": float(np.percentile(arr_flat, 99)),
        "nonzero_ratio": float((arr_flat > 0).mean()),
    }

    return stats


case_dirs = sorted(os.listdir(DATA_ROOT))

all_stats = []

for case_id in tqdm(case_dirs):
    pet_path = os.path.join(DATA_ROOT, case_id, f"{case_id}__PT.nii.gz")
    if not os.path.exists(pet_path):
        continue

    stats = analyze_pet(pet_path)
    stats["case"] = case_id
    all_stats.append(stats)

# 打印整体统计
for key in ["p50", "p90", "p95", "p99"]:
    vals = [s[key] for s in all_stats]
    print(f"{key}: mean={np.mean(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}")