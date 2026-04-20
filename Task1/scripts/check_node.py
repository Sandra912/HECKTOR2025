import os
import json
import numpy as np

# ====== 配置 ======
data_root = "/data/HECKTOR2025/Task1"
# label_dir = "labelsTr_tune_npy"
label_dir = "labelsTr_resampled_npy"
# split_file = "/home/aims/projects/HECKTOR2025/Task1/config/splits_tune_train30_valexisting.json"
split_file = "/home/aims/projects/HECKTOR2025/Task1/config/splits_available.json"


# ====== 读取 split ======
with open(split_file, "r") as f:
    splits = json.load(f)

fold = splits[0]
train_ids = fold["train"]
val_ids = fold["val"]


def load_label(path):
    data = np.load(path)
    # 自动取第一个 key
    key = data.files[0]
    return data[key]


def check_cases(case_ids, name):
    total = len(case_ids)
    has_node = 0
    no_node = 0
    missing = 0

    has_node_cases = []

    for cid in case_ids:
        label_path = os.path.join(data_root, label_dir, f"{cid}_label.npz")

        if not os.path.exists(label_path):
            missing += 1
            continue

        try:
            arr = load_label(label_path)
        except Exception as e:
            print(f"[ERROR] load failed: {cid} -> {e}")
            missing += 1
            continue

        unique_vals = np.unique(arr)

        if 2 in unique_vals:
            has_node += 1
            has_node_cases.append(cid)
        else:
            no_node += 1

    print(f"\n===== fold0 {name} =====")
    print(f"{name} total = {total}")
    print(f"{name} with node (label==2) = {has_node}")
    print(f"{name} without node = {no_node}")
    print(f"{name} missing label file = {missing}")

    print(f"\n{name} cases WITH node (first 20):")
    for cid in has_node_cases[:20]:
        print(cid)


# ====== 执行 ======
check_cases(train_ids, "train")
check_cases(val_ids, "val")