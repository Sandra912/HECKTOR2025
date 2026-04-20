import os
import time
import SimpleITK as sitk
import pandas as pd

data_root = "/data/HECKTOR2025/Task1"   # 改成你的原始数据目录

rows = []

case_ids = sorted(os.listdir(data_root))
print(f"Found {len(case_ids)} entries under {data_root}")

for i, case_id in enumerate(case_ids, 1):
    case_dir = os.path.join(data_root, case_id)
    if not os.path.isdir(case_dir):
        continue

    print(f"\n[{i}/{len(case_ids)}] Processing {case_id}")
    t0 = time.time()

    ct_path = os.path.join(case_dir, f"{case_id}__CT.nii.gz")
    pet_path = os.path.join(case_dir, f"{case_id}__PT.nii.gz")

    if os.path.exists(ct_path):
        print(f"  Reading CT: {ct_path}")
        ct_img = sitk.ReadImage(ct_path)
        print(f"  CT spacing: {ct_img.GetSpacing()}")
        rows.append({
            "case_id": case_id,
            "modality": "CT",
            "sx": ct_img.GetSpacing()[0],
            "sy": ct_img.GetSpacing()[1],
            "sz": ct_img.GetSpacing()[2],
        })
    else:
        print("  CT not found")

    if os.path.exists(pet_path):
        print(f"  Reading PET: {pet_path}")
        pet_img = sitk.ReadImage(pet_path)
        print(f"  PET spacing: {pet_img.GetSpacing()}")
        rows.append({
            "case_id": case_id,
            "modality": "PET",
            "sx": pet_img.GetSpacing()[0],
            "sy": pet_img.GetSpacing()[1],
            "sz": pet_img.GetSpacing()[2],
        })
    else:
        print("  PET not found")

    print(f"  Done in {time.time() - t0:.2f}s")

df = pd.DataFrame(rows)

print("\n========== CT spacing stats ==========")
print(df[df["modality"] == "CT"][["sx", "sy", "sz"]].describe())

print("\n========== PET spacing stats ==========")
print(df[df["modality"] == "PET"][["sx", "sy", "sz"]].describe())

print("\n========== CT quantiles ==========")
print(df[df["modality"] == "CT"][["sx", "sy", "sz"]].quantile([0.25, 0.5, 0.75]))

print("\n========== PET quantiles ==========")
print(df[df["modality"] == "PET"][["sx", "sy", "sz"]].quantile([0.25, 0.5, 0.75]))