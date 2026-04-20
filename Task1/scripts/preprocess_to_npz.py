#!/usr/bin/env python3
"""
Parallel preprocessing for HECKTOR 2025 Task1
(PET-only + head-neck aware + tumor-safe crop)

Input:
  /data/HECKTOR2025/Task1/CASE_ID/
    CASE_ID__PT.nii.gz
    CASE_ID.nii.gz

Output:
  /data/HECKTOR2025/Task1/imagesTr_tune_npy/CASE_ID_pet.npz
  /data/HECKTOR2025/Task1/labelsTr_tune_npy/CASE_ID_label.npz

Pipeline
1. Load PET & label
2. Reorient → LPS
3. Resample
   - Target spacing: (4.0, 4.0, 3.0) mm
   - PET: linear interpolation
   - Label: nearest neighbor (aligned to PET grid)
4. PET-based bbox
   - Smooth: Gaussian σ = 1.0
   - Search region: top 60% (z ≥ 0.40)
   - Threshold: 85th percentile (≥ 0.0)
   - Morphology: opening + closing (3×3×3)
   - Remove small CC: < 500 voxels
   - Select best component (size + intensity + higher z)
5. Label bbox
   - Foreground: labels {1, 2}
6. Final crop
   bbox = union(PET_bbox, label_bbox)
   - Padding: (6, 6, 6) voxels
   - Clip to image boundary
7. Crop & save
   - Shape: [1, X, Y, Z]
   - PET: float16
   - Label: uint8
   - Save as .npz with metadata

"""

import os
import json
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, label as cc_label


# =========================
# User settings
# =========================
DATA_ROOT = "/data/HECKTOR2025/Task1"

# IMAGES_OUT = os.path.join(DATA_ROOT, "imagesTr_tune_npy")
# LABELS_OUT = os.path.join(DATA_ROOT, "labelsTr_tune_npy")
IMAGES_OUT = os.path.join(DATA_ROOT, "imagesTr_resampled_npy")
LABELS_OUT = os.path.join(DATA_ROOT, "labelsTr_resampled_npy")


# SPLITS_FILE = "/home/aims/projects/HECKTOR2025/Task1/config/splits_tune_train30_valexisting.json"
SPLITS_FILE = "/home/aims/projects/HECKTOR2025/Task1/config/splits_available.json"

NUM_WORKERS = 4
TARGET_SPACING = (4.0, 4.0, 3.0)  # (X, Y, Z) mm
VALID_LABELS = {0, 1, 2}

FOLD_IDX = 0
SPLIT_MODE = "both"   # "train" / "val" / "both" / "all_folds"

# Crop settings
PET_CROP_PERCENTILE = 85.0
PET_CROP_MIN_THRESHOLD = 0.0
HEAD_NECK_Z_START_FRACTION = 0.40   # only search connected components in upper 60% of volume
SMOOTH_SIGMA = 1.0
MIN_CC_VOXELS = 500
CROP_PADDING_VOXELS = (6, 6, 6)     # (X, Y, Z)


# =========================
# Utility functions
# =========================
def load_case_ids_from_splits(splits_file: str, fold_idx: int = 0, split_mode: str = "train"):
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError("Splits file must contain a non-empty list of folds.")

    if split_mode not in {"train", "val", "both", "all_folds"}:
        raise ValueError("split_mode must be one of {'train', 'val', 'both', 'all_folds'}.")

    if split_mode == "all_folds":
        case_ids = set()
        for fold in splits:
            case_ids.update(fold["train"])
            case_ids.update(fold["val"])
        return sorted(case_ids)

    if not (0 <= fold_idx < len(splits)):
        raise ValueError(f"fold_idx={fold_idx} out of range. splits has {len(splits)} folds.")

    fold = splits[fold_idx]

    if split_mode == "train":
        return sorted(fold["train"])
    elif split_mode == "val":
        return sorted(fold["val"])
    else:
        return sorted(set(fold["train"]) | set(fold["val"]))


def build_meta(img_sitk: sitk.Image):
    return {
        "spacing": tuple(float(x) for x in img_sitk.GetSpacing()),
        "origin": tuple(float(x) for x in img_sitk.GetOrigin()),
        "direction": tuple(float(x) for x in img_sitk.GetDirection()),
        "size": tuple(int(x) for x in img_sitk.GetSize()),  # (X, Y, Z)
        "dimension": int(img_sitk.GetDimension()),
        "pixel_id": int(img_sitk.GetPixelID()),
        "pixel_type": str(img_sitk.GetPixelIDTypeAsString()),
    }


def save_npz(path: str, image: np.ndarray, meta: dict):
    np.savez_compressed(path, image=image, meta=meta)


def reorient_to_lps(img: sitk.Image) -> sitk.Image:
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(img)


def compute_new_size(old_size, old_spacing, new_spacing):
    new_size = [
        int(np.round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(old_size, old_spacing, new_spacing)
    ]
    return [max(1, x) for x in new_size]


def resample_image_to_spacing(
    image: sitk.Image,
    out_spacing=(4.0, 4.0, 3.0),
    is_label: bool = False,
    default_value: float = 0.0,
) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    out_size = compute_new_size(original_size, original_spacing, out_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize([int(x) for x in out_size])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    is_label: bool = False,
    default_value: float = 0.0,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def sitk_to_numpy_channel_first(img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)      # [Z, Y, X]
    arr = np.transpose(arr, (2, 1, 0))     # [X, Y, Z]
    arr = np.expand_dims(arr, axis=0)      # [1, X, Y, Z]
    return arr


def check_finite(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def check_label_values(label: np.ndarray):
    uniq = set(np.unique(label).tolist())
    if not uniq.issubset(VALID_LABELS):
        raise ValueError(
            f"Invalid label values found: {sorted(uniq)}. "
            f"Expected subset of {sorted(VALID_LABELS)}."
        )


def maybe_warn_empty_label(label: np.ndarray):
    uniq = np.unique(label)
    return len(uniq) == 1 and uniq[0] == 0


def bbox_from_binary_mask_zyx(mask_zyx: np.ndarray):
    if not np.any(mask_zyx):
        return None

    zz, yy, xx = np.where(mask_zyx)
    return {
        "x_start": int(xx.min()),
        "x_end": int(xx.max()) + 1,
        "y_start": int(yy.min()),
        "y_end": int(yy.max()) + 1,
        "z_start": int(zz.min()),
        "z_end": int(zz.max()) + 1,
    }


def union_bbox(b1: dict, b2: dict):
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    return {
        "x_start": min(b1["x_start"], b2["x_start"]),
        "x_end": max(b1["x_end"], b2["x_end"]),
        "y_start": min(b1["y_start"], b2["y_start"]),
        "y_end": max(b1["y_end"], b2["y_end"]),
        "z_start": min(b1["z_start"], b2["z_start"]),
        "z_end": max(b1["z_end"], b2["z_end"]),
    }


def pad_and_clip_bbox(bbox: dict, full_size_xyz, padding_voxels=(6, 6, 6)):
    if bbox is None:
        sx, sy, sz = full_size_xyz
        return {
            "x_start": 0, "x_end": sx,
            "y_start": 0, "y_end": sy,
            "z_start": 0, "z_end": sz,
        }

    pad_x, pad_y, pad_z = padding_voxels
    size_x, size_y, size_z = full_size_xyz

    x0 = max(0, bbox["x_start"] - pad_x)
    x1 = min(size_x, bbox["x_end"] + pad_x)
    y0 = max(0, bbox["y_start"] - pad_y)
    y1 = min(size_y, bbox["y_end"] + pad_y)
    z0 = max(0, bbox["z_start"] - pad_z)
    z1 = min(size_z, bbox["z_end"] + pad_z)

    return {
        "x_start": x0, "x_end": x1,
        "y_start": y0, "y_end": y1,
        "z_start": z0, "z_end": z1,
    }


def compute_head_neck_pet_bbox(
    pet_img_resampled: sitk.Image,
    percentile: float = 85.0,
    min_threshold: float = 0.0,
    z_start_fraction: float = 0.40,
    smooth_sigma: float = 1.0,
    min_cc_voxels: int = 500,
):
    """
    Head-neck aware PET bbox:
      1) smooth PET
      2) restrict connected-component search to upper part of volume
      3) threshold in that region
      4) morphology cleanup
      5) connected components
      6) choose best component by score favoring upper-volume location
    """
    pet_arr_zyx = sitk.GetArrayFromImage(pet_img_resampled).astype(np.float32)  # [Z, Y, X]
    pet_smooth = gaussian_filter(pet_arr_zyx, sigma=smooth_sigma)

    z_size = pet_smooth.shape[0]
    z0_search = int(z_size * z_start_fraction)
    candidate_region = pet_smooth[z0_search:, :, :]

    threshold = max(float(min_threshold), float(np.percentile(candidate_region, percentile)))

    mask_zyx = pet_smooth > threshold

    # Only use upper part for selecting candidate connected components
    search_mask_zyx = np.zeros_like(mask_zyx, dtype=bool)
    search_mask_zyx[z0_search:, :, :] = mask_zyx[z0_search:, :, :]

    structure = np.ones((3, 3, 3), dtype=bool)
    search_mask_zyx = binary_opening(search_mask_zyx, structure=structure)
    search_mask_zyx = binary_closing(search_mask_zyx, structure=structure)

    labeled, num = cc_label(search_mask_zyx, structure=np.ones((3, 3, 3), dtype=np.uint8))

    if num == 0:
        return None, threshold

    components = []
    for comp_id in range(1, num + 1):
        comp_mask = labeled == comp_id
        comp_size = int(comp_mask.sum())
        if comp_size < min_cc_voxels:
            continue

        zz, yy, xx = np.where(comp_mask)
        z_center = float(np.mean(zz))
        mean_intensity = float(np.mean(pet_smooth[comp_mask]))

        # Score: favor larger, brighter, and more superior components
        # (superior = larger z index here because we search in upper slices)
        score = (0.001 * comp_size) + mean_intensity + (0.01 * z_center)

        components.append((score, comp_mask))

    if len(components) == 0:
        return None, threshold

    components.sort(key=lambda x: x[0], reverse=True)
    best_mask = components[0][1]
    best_bbox = bbox_from_binary_mask_zyx(best_mask)

    return best_bbox, threshold


def crop_sitk_with_bbox(img: sitk.Image, bbox_xyz: dict) -> sitk.Image:
    x0, x1 = bbox_xyz["x_start"], bbox_xyz["x_end"]
    y0, y1 = bbox_xyz["y_start"], bbox_xyz["y_end"]
    z0, z1 = bbox_xyz["z_start"], bbox_xyz["z_end"]

    index = [int(x0), int(y0), int(z0)]
    size = [int(x1 - x0), int(y1 - y0), int(z1 - z0)]

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(index)
    roi.SetSize(size)
    return roi.Execute(img)


def bbox_to_list_xyz(bbox_xyz: dict):
    return [
        int(bbox_xyz["x_start"]), int(bbox_xyz["x_end"]),
        int(bbox_xyz["y_start"]), int(bbox_xyz["y_end"]),
        int(bbox_xyz["z_start"]), int(bbox_xyz["z_end"]),
    ]


def make_modality_meta(
    case_id: str,
    modality: str,
    raw_path: str,
    target_spacing,
    raw_meta: dict,
    lps_meta: dict,
    pre_crop_resampled_meta: dict,
    cropped_resampled_meta: dict,
    extra: dict = None,
):
    meta = {
        "case_id": case_id,
        "modality": modality,
        "raw_path": raw_path,
        "target_spacing": tuple(float(x) for x in target_spacing),
        "raw_meta": raw_meta,
        "lps_meta": lps_meta,
        "pre_crop_resampled_meta": pre_crop_resampled_meta,
        "cropped_resampled_meta": cropped_resampled_meta,
    }
    if extra is not None:
        meta.update(extra)
    return meta


# =========================
# Main case processing
# =========================
def process_case(case_id: str):
    try:
        case_dir = os.path.join(DATA_ROOT, case_id)

        pet_path = os.path.join(case_dir, f"{case_id}__PT.nii.gz")
        label_path = os.path.join(case_dir, f"{case_id}.nii.gz")

        if not (os.path.exists(pet_path) and os.path.exists(label_path)):
            return ("missing", case_id, "PET or label file does not exist")

        # -------------------------
        # Read raw images
        # -------------------------
        pet_img_raw = sitk.ReadImage(pet_path)
        label_img_raw = sitk.ReadImage(label_path)

        raw_pet_meta = build_meta(pet_img_raw)
        raw_label_meta = build_meta(label_img_raw)

        # -------------------------
        # Reorient to LPS
        # -------------------------
        pet_img_lps = reorient_to_lps(pet_img_raw)
        label_img_lps = reorient_to_lps(label_img_raw)

        lps_pet_meta = build_meta(pet_img_lps)
        lps_label_meta = build_meta(label_img_lps)

        # -------------------------
        # Resample PET to fixed spacing
        # -------------------------
        pet_img_resampled_full = resample_image_to_spacing(
            pet_img_lps,
            out_spacing=TARGET_SPACING,
            is_label=False,
            default_value=0.0,
        )

        # -------------------------
        # Resample label onto PET grid
        # -------------------------
        label_img_resampled_full = resample_to_reference(
            label_img_lps,
            pet_img_resampled_full,
            is_label=True,
            default_value=0,
        )

        pre_crop_pet_meta = build_meta(pet_img_resampled_full)
        pre_crop_label_meta = build_meta(label_img_resampled_full)

        full_size_xyz = tuple(int(x) for x in pre_crop_pet_meta["size"])

        # -------------------------
        # Head-neck aware PET bbox
        # -------------------------
        pet_bbox, pet_crop_threshold = compute_head_neck_pet_bbox(
            pet_img_resampled=pet_img_resampled_full,
            percentile=PET_CROP_PERCENTILE,
            min_threshold=PET_CROP_MIN_THRESHOLD,
            z_start_fraction=HEAD_NECK_Z_START_FRACTION,
            smooth_sigma=SMOOTH_SIGMA,
            min_cc_voxels=MIN_CC_VOXELS,
        )

        # -------------------------
        # Tumor-safe label bbox
        # -------------------------
        label_arr_zyx = sitk.GetArrayFromImage(label_img_resampled_full).astype(np.uint8)
        label_bbox = bbox_from_binary_mask_zyx(label_arr_zyx > 0)

        # If PET bbox exists, union with label bbox.
        # If PET bbox fails, label bbox alone still guarantees tumor-safe crop.
        merged_bbox = union_bbox(pet_bbox, label_bbox)

        # Final fallback: full image
        final_bbox = pad_and_clip_bbox(
            merged_bbox,
            full_size_xyz=full_size_xyz,
            padding_voxels=CROP_PADDING_VOXELS,
        )

        # -------------------------
        # Crop PET and label
        # -------------------------
        pet_img_cropped = crop_sitk_with_bbox(pet_img_resampled_full, final_bbox)
        label_img_cropped = crop_sitk_with_bbox(label_img_resampled_full, final_bbox)

        cropped_pet_meta = build_meta(pet_img_cropped)
        cropped_label_meta = build_meta(label_img_cropped)

        # -------------------------
        # Convert to numpy
        # -------------------------
        pet = sitk_to_numpy_channel_first(pet_img_cropped).astype(np.float16)
        label = sitk_to_numpy_channel_first(label_img_cropped).astype(np.uint8)

        if pet.shape != label.shape:
            return (
                "error",
                case_id,
                f"Shape mismatch after crop: pet={pet.shape}, label={label.shape}",
            )

        check_finite("PET", pet)
        check_label_values(label)

        empty_label = maybe_warn_empty_label(label)

        crop_extra = {
            "crop_applied": True,
            "crop_source": "head-neck aware PET connected component + tumor-safe label union",
            "pet_crop_percentile": float(PET_CROP_PERCENTILE),
            "pet_crop_min_threshold": float(PET_CROP_MIN_THRESHOLD),
            "pet_crop_threshold_used": float(pet_crop_threshold),
            "head_neck_z_start_fraction": float(HEAD_NECK_Z_START_FRACTION),
            "smooth_sigma": float(SMOOTH_SIGMA),
            "min_cc_voxels": int(MIN_CC_VOXELS),
            "crop_padding_voxels_xyz": tuple(int(x) for x in CROP_PADDING_VOXELS),
            "pet_bbox_xyz": bbox_to_list_xyz(pet_bbox) if pet_bbox is not None else None,
            "label_bbox_xyz": bbox_to_list_xyz(label_bbox) if label_bbox is not None else None,
            "crop_bbox_format": "start_inclusive_end_exclusive_xyz_on_pre_crop_resampled_grid",
            "crop_bbox_xyz": bbox_to_list_xyz(final_bbox),
            "full_resampled_size_xyz": tuple(int(x) for x in pre_crop_pet_meta["size"]),
            "cropped_size_xyz": tuple(int(x) for x in cropped_pet_meta["size"]),
        }

        # -------------------------
        # Build metadata
        # -------------------------
        pet_meta = make_modality_meta(
            case_id=case_id,
            modality="PET",
            raw_path=pet_path,
            target_spacing=TARGET_SPACING,
            raw_meta=raw_pet_meta,
            lps_meta=lps_pet_meta,
            pre_crop_resampled_meta=pre_crop_pet_meta,
            cropped_resampled_meta=cropped_pet_meta,
            extra=crop_extra,
        )

        label_meta = make_modality_meta(
            case_id=case_id,
            modality="LABEL",
            raw_path=label_path,
            target_spacing=TARGET_SPACING,
            raw_meta=raw_label_meta,
            lps_meta=lps_label_meta,
            pre_crop_resampled_meta=pre_crop_label_meta,
            cropped_resampled_meta=cropped_label_meta,
            extra={
                **crop_extra,
                "valid_labels": sorted(list(VALID_LABELS)),
            },
        )

        # -------------------------
        # Save
        # -------------------------
        pet_out = os.path.join(IMAGES_OUT, f"{case_id}_pet.npz")
        label_out = os.path.join(LABELS_OUT, f"{case_id}_label.npz")

        save_npz(pet_out, pet, pet_meta)
        save_npz(label_out, label, label_meta)

        msg = (
            f"saved cropped_shape={pet.shape}, "
            f"target_spacing={TARGET_SPACING}, "
            f"raw_pet_size={raw_pet_meta['size']}, "
            f"full_resampled_pet_size={pre_crop_pet_meta['size']}, "
            f"cropped_pet_size={cropped_pet_meta['size']}, "
            f"pet_crop_threshold_used={pet_crop_threshold:.6f}, "
            f"pet_bbox_xyz={bbox_to_list_xyz(pet_bbox) if pet_bbox is not None else None}, "
            f"label_bbox_xyz={bbox_to_list_xyz(label_bbox) if label_bbox is not None else None}, "
            f"crop_bbox_xyz={bbox_to_list_xyz(final_bbox)}, "
            f"label_values={sorted(np.unique(label).tolist())}"
        )

        if empty_label:
            return ("warning", case_id, msg + " | label is all background")

        return ("ok", case_id, msg)

    except Exception:
        return ("error", case_id, traceback.format_exc())


# =========================
# Main
# =========================
def main():
    os.makedirs(IMAGES_OUT, exist_ok=True)
    os.makedirs(LABELS_OUT, exist_ok=True)

    case_ids = load_case_ids_from_splits(
        SPLITS_FILE,
        fold_idx=FOLD_IDX,
        split_mode=SPLIT_MODE,
    )
    worker_count = min(NUM_WORKERS, cpu_count())

    print(f"Splits file: {SPLITS_FILE}")
    print(f"Split mode: {SPLIT_MODE}")
    if SPLIT_MODE != "all_folds":
        print(f"Fold index: {FOLD_IDX}")
    print(f"Found {len(case_ids)} case IDs from splits file.")
    print(f"Using {worker_count} worker processes.")
    print(f"Target spacing: {TARGET_SPACING}")
    print(f"PET crop percentile: {PET_CROP_PERCENTILE}")
    print(f"PET crop min threshold: {PET_CROP_MIN_THRESHOLD}")
    print(f"Head-neck z start fraction: {HEAD_NECK_Z_START_FRACTION}")
    print(f"Smooth sigma: {SMOOTH_SIGMA}")
    print(f"Min connected component voxels: {MIN_CC_VOXELS}")
    print(f"Crop padding voxels XYZ: {CROP_PADDING_VOXELS}")
    print(f"Image output dir: {IMAGES_OUT}")
    print(f"Label output dir: {LABELS_OUT}")

    ok_count = 0
    warning_count = 0
    missing_count = 0
    error_count = 0

    warning_cases = []
    missing_cases = []
    error_cases = []

    chunksize = 1

    with Pool(processes=worker_count) as pool:
        pbar = tqdm(
            total=len(case_ids),
            desc="Preprocessing cases",
            unit="case",
            dynamic_ncols=True,
        )

        for status, case_id, msg in pool.imap_unordered(
            process_case, case_ids, chunksize=chunksize
        ):
            if status == "ok":
                ok_count += 1
            elif status == "warning":
                warning_count += 1
                warning_cases.append((case_id, msg))
                print(f"\n[Warning] {case_id}")
                print(msg)
            elif status == "missing":
                missing_count += 1
                missing_cases.append(case_id)
                print(f"\n[Missing] {case_id}")
            else:
                error_count += 1
                error_cases.append((case_id, msg))
                print(f"\n[Error] {case_id}")
                print(msg)

            pbar.update(1)
            pbar.set_postfix(
                ok=ok_count,
                warning=warning_count,
                missing=missing_count,
                error=error_count,
            )

        pbar.close()

    print("\n=== Done ===")
    print(f"Converted cases: {ok_count}")
    print(f"Warning cases:   {warning_count}")
    print(f"Missing cases:   {missing_count}")
    print(f"Error cases:     {error_count}")

    if warning_cases:
        print("\nFirst warning cases:")
        for cid, msg in warning_cases[:10]:
            print(f"  {cid}: {msg}")

    if missing_cases:
        print("\nFirst missing cases:")
        for cid in missing_cases[:20]:
            print(" ", cid)

    if error_cases:
        print("\nFirst error cases:")
        for cid, msg in error_cases[:5]:
            print(f"\n--- {cid} ---")
            print(msg)


if __name__ == "__main__":
    main()