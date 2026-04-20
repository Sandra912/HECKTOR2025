#!/usr/bin/env python3
"""
Inference script for HECKTOR 2025 Task 1 segmentation models (PET-only version).

Official Task 1 alignment:
1. Final exported mask is saved in the ORIGINAL CT grid (same size/spacing/origin/direction as CT)
2. Exported label values are strictly {0, 1, 2}
3. One final NIfTI mask per case for submission
4. Internal evaluation can be done either in resampled space or original CT space

This version is aligned with PET-only preprocessing:
- supports cropped resampled PET NPZ
- restores cropped prediction back to full resampled PET grid
- resamples restored prediction to original CT grid
- supports GTVn postprocessing (remove tiny connected components)
- supports lesion-aware metrics for GTVn
- applies PET intensity preprocessing consistent with training:
    clamp to [0, 30], then scale to [0, 1]
"""

"""
Example:
python ./scripts/inference.py \
  --model_path /home/aims/projects/HECKTOR2025/Task1/outputs/unet3d/fold_0/checkpoints/best_model.pth \
  --pet_path /data/HECKTOR2025/Task1/imagesTr_resampled_npy/CHUS-027_pet.npz \
  --label_path /data/HECKTOR2025/Task1/labelsTr_resampled_npy/CHUS-027_label.npz \
  --original_ct_path /data/HECKTOR2025/Task1/CHUS-027/CHUS-027__CT.nii.gz \
  --original_label_path /data/HECKTOR2025/Task1/CHUS-027/CHUS-027.nii.gz \
  --device cuda \
  --min-gtvn-size 10
"""

import os
import sys
import csv
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import label as cc_label
from monai.inferers import sliding_window_inference

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

try:
    from models import UNet3DModel, SegResNetModel, UNETRModel, SwinUNETRModel
    from config import UNet3DConfig, SegResNetConfig, UNETRConfig, SwinUNETRConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the Task1 directory")
    sys.exit(1)


VALID_LABELS = {0, 1, 2}
GTVN_IOU_THRESHOLD = 0.30

# Must match training transforms
PET_CLIP_MIN = 0.0
PET_CLIP_MAX = 30.0


def load_npz_payload(npz_path: str) -> Dict:
    loaded = np.load(npz_path, allow_pickle=True)
    payload = {}
    for k in loaded.files:
        value = loaded[k]
        if k == "meta":
            if isinstance(value, np.ndarray) and value.shape == ():
                payload[k] = value.item()
            else:
                payload[k] = value
        else:
            payload[k] = value
    return payload


def binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    pred_sum = int(pred_mask.sum())
    gt_sum = int(gt_mask.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int(np.logical_and(pred_mask, gt_mask).sum())
    return float(2.0 * intersection / (pred_sum + gt_sum))


def connected_components_3d(mask: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    """26-connectivity connected components for 3D binary mask."""
    mask = mask.astype(np.uint8)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num = cc_label(mask, structure=structure)

    components = []
    sizes = []
    for comp_id in range(1, num + 1):
        comp = (labeled == comp_id)
        components.append(comp)
        sizes.append(int(comp.sum()))
    return components, sizes


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size voxels."""
    if min_size <= 0:
        return mask.astype(bool)

    components, sizes = connected_components_3d(mask)
    out = np.zeros_like(mask, dtype=bool)
    for comp, size in zip(components, sizes):
        if size >= min_size:
            out |= comp
    return out


def lesion_detection_counts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = GTVN_IOU_THRESHOLD,
) -> Tuple[int, int, int]:
    """
    Lesion-level matching for GTVn:
    TP if IoU > threshold with one-to-one greedy matching.
    Returns: (tp, fp, fn)
    """
    pred_components, pred_sizes = connected_components_3d(pred_mask)
    gt_components, gt_sizes = connected_components_3d(gt_mask)

    if len(pred_components) == 0 and len(gt_components) == 0:
        return 0, 0, 0
    if len(pred_components) == 0:
        return 0, 0, len(gt_components)
    if len(gt_components) == 0:
        return 0, len(pred_components), 0

    candidates = []
    for i, pred_comp in enumerate(pred_components):
        pred_size = pred_sizes[i]
        for j, gt_comp in enumerate(gt_components):
            inter = int(np.logical_and(pred_comp, gt_comp).sum())
            if inter == 0:
                continue
            union = pred_size + gt_sizes[j] - inter
            iou = inter / union
            if iou > iou_threshold:
                candidates.append((iou, inter, i, j))

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))

    matched_pred = set()
    matched_gt = set()
    tp = 0

    for _, _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        tp += 1

    fp = len(pred_components) - tp
    fn = len(gt_components) - tp
    return tp, fp, fn


def lesion_aware_aggregated_dsc_parts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Lesion-aware aggregated DSC parts for GTVn.
    Returns:
        matched_intersection_sum, pred_sum_total, gt_sum_total
    """
    pred_components, pred_sizes = connected_components_3d(pred_mask)
    gt_components, gt_sizes = connected_components_3d(gt_mask)

    pred_sum_total = int(np.sum(pred_sizes))
    gt_sum_total = int(np.sum(gt_sizes))

    if len(pred_components) == 0 and len(gt_components) == 0:
        return 0, 0, 0
    if len(pred_components) == 0 or len(gt_components) == 0:
        return 0, pred_sum_total, gt_sum_total

    candidates = []
    for i, pred_comp in enumerate(pred_components):
        pred_size = pred_sizes[i]
        for j, gt_comp in enumerate(gt_components):
            inter = int(np.logical_and(pred_comp, gt_comp).sum())
            if inter == 0:
                continue
            union = pred_size + gt_sizes[j] - inter
            iou = inter / union
            candidates.append((iou, inter, i, j))

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))

    matched_pred = set()
    matched_gt = set()
    matched_intersection_sum = 0

    for _, inter, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matched_intersection_sum += inter

    return matched_intersection_sum, pred_sum_total, gt_sum_total


def check_prediction_values(prediction: np.ndarray, name: str = "Prediction"):
    uniq = set(np.unique(prediction).tolist())
    if not uniq.issubset(VALID_LABELS):
        raise ValueError(
            f"{name} has invalid labels: {sorted(uniq)}. Expected subset of {sorted(VALID_LABELS)}."
        )


def postprocess_prediction_task1(
    pred: np.ndarray,
    min_gtvn_size: int = 10,
) -> np.ndarray:
    """
    Task 1 postprocessing for inference / validation:
    - keep GTVp as-is
    - remove tiny GTVn connected components
    - enforce label space {0,1,2}
    """
    check_prediction_values(pred, "Prediction before postprocess")

    out = np.zeros_like(pred, dtype=np.uint8)
    out[pred == 1] = 1

    pred_n = (pred == 2)
    pred_n = remove_small_components(pred_n, min_gtvn_size)

    pred_n = np.logical_and(pred_n, out != 1)
    out[pred_n] = 2

    check_prediction_values(out, "Prediction after postprocess")
    return out


def preprocess_pet_for_model(pet_data_xyz: np.ndarray) -> np.ndarray:
    """
    Must match training intensity preprocessing:
      1) clamp to >= 0
      2) clip to <= 30
      3) scale to [0, 1]
    Input/output shape: [X, Y, Z]
    """
    pet = pet_data_xyz.astype(np.float32)
    pet = np.clip(pet, PET_CLIP_MIN, PET_CLIP_MAX)
    pet = pet / PET_CLIP_MAX
    return pet


def load_npz_data(pet_path: str):
    """Load PET data and PET meta from NPZ, return model-ready tensor."""
    print(f"Loading PET data from: {pet_path}")
    pet_payload = load_npz_payload(pet_path)
    pet_data = pet_payload["image"]
    pet_meta = pet_payload.get("meta", None)

    print(f"Raw PET shape: {pet_data.shape}")

    if pet_data.ndim != 4:
        raise ValueError(f"Expected PET shape [1, X, Y, Z], got PET={pet_data.shape}")

    if pet_data.shape[0] == 1:
        pet_data = pet_data.squeeze(0)

    print(f"After squeeze - PET shape: {pet_data.shape}")

    pet_model = preprocess_pet_for_model(pet_data)  # [X, Y, Z]
    image = np.expand_dims(pet_model, axis=0)       # [1, X, Y, Z]
    tensor = torch.from_numpy(image).float().unsqueeze(0)  # [1, 1, X, Y, Z]

    return tensor, pet_data, pet_meta


def load_label_data(label_path: str):
    """Load label NPZ. Return label [X, Y, Z] and optional meta."""
    print(f"Loading label data from: {label_path}")
    payload = load_npz_payload(label_path)
    label = payload["image"]
    label_meta = payload.get("meta", None)

    print(f"Raw label shape: {label.shape}")

    if label.ndim != 4:
        raise ValueError(f"Expected label shape [1, X, Y, Z], got {label.shape}")

    if label.shape[0] == 1:
        label = label.squeeze(0)

    label = label.astype(np.int16)
    print(f"After squeeze - label shape: {label.shape}")
    return label, label_meta


def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "model_config" not in checkpoint:
            print("Error: No model_config found in checkpoint")
            return None, None

        model_config = checkpoint["model_config"]
        experiment_name = model_config.get("experiment_name", "unet3d")
        print(f"Found model config for experiment: {experiment_name}")

        if experiment_name == "unet3d":
            config = UNet3DConfig()
        elif experiment_name == "segresnet":
            config = SegResNetConfig()
        elif experiment_name == "unetr":
            config = UNETRConfig()
        elif experiment_name == "swinunetr":
            config = SwinUNETRConfig()
        else:
            print(f"Error: Unknown experiment type: {experiment_name}")
            return None, None

        for key, value in model_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Safety: force PET-only input channel
        config.input_channels = 1

        print(f"Recreated config for {experiment_name}")
        print(f"  Input channels: {config.input_channels}")
        print(f"  Num classes: {config.num_classes}")
        print(f"  Spatial size: {config.spatial_size}")
        print(f"  Fold: {config.fold}")

        if experiment_name == "unet3d":
            model = UNet3DModel(config)
        elif experiment_name == "segresnet":
            model = SegResNetModel(config)
        elif experiment_name == "unetr":
            model = UNETRModel(config)
        elif experiment_name == "swinunetr":
            model = SwinUNETRModel(config)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        print("✓ Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, config

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_inference(model, input_tensor, config, device="cuda", use_sliding_window=False):
    """Run inference and return raw prediction as numpy array with shape [X, Y, Z]."""
    print(f"Input tensor shape: {input_tensor.shape}")

    try:
        input_tensor = input_tensor.to(device)

        if use_sliding_window:
            print("Running sliding window inference...")
            print(f"  ROI size: {config.spatial_size}")
            print("  SW batch size: 4")
            print("  Overlap: 0.5")

            with torch.no_grad():
                output = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=config.spatial_size,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0,
                    sw_device=device,
                    device=device,
                )
        else:
            print("Running direct inference...")
            with torch.no_grad():
                output = model(input_tensor)

        print(f"Output shape: {output.shape}")
        print(f"Output min/max: {output.min().item():.4f} / {output.max().item():.4f}")

        if output.shape[1] > 1:
            probs = torch.softmax(output, dim=1)
            prediction = probs.argmax(dim=1)
            print(f"Using softmax + argmax for {output.shape[1]} classes")
        else:
            probs = torch.sigmoid(output)
            prediction = (probs > 0.5).long()
            print("Using sigmoid for binary classification")

        print(f"Prediction shape before squeeze: {prediction.shape}")

        unique_vals = torch.unique(prediction).tolist()
        print(f"Unique values in raw prediction: {unique_vals}")

        total_voxels = prediction.numel()
        for val in unique_vals:
            count = (prediction == val).sum().item()
            percentage = (count / total_voxels) * 100
            print(f"  Class {val}: {count:,} voxels ({percentage:.6f}%)")

        prediction_np = prediction.cpu().numpy().squeeze(0).astype(np.uint8)  # [X, Y, Z]
        print(f"Prediction shape after squeeze: {prediction_np.shape}")

        check_prediction_values(prediction_np, "Raw prediction")
        return prediction_np

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_single_case(prediction: np.ndarray, label: np.ndarray) -> Dict:
    """
    Task1-oriented single-case metrics aligned with updated train.py:
    - GTVp Dice
    - GTVn lesion-aware aggregated DSC parts
    - GTVn detection TP/FP/FN
    """
    result = {
        "shape_match": prediction.shape == label.shape,
        "pred_unique": np.unique(prediction).tolist(),
        "label_unique": np.unique(label).tolist(),
        "gtvp_dsc": np.nan,
        "gtvn_case_binary_dsc": np.nan,
        "gtvn_matched_intersection": 0,
        "gtvn_pred_voxels": 0,
        "gtvn_gt_voxels": 0,
        "gtvn_agg_dsc_case_proxy": np.nan,
        "gtvn_tp": 0,
        "gtvn_fp": 0,
        "gtvn_fn": 0,
        "gtvn_f1_case": np.nan,
    }

    print("\n=== Prediction vs Label Check ===")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Label shape:      {label.shape}")
    print(f"Shape match:      {result['shape_match']}")
    print(f"Prediction unique values: {result['pred_unique']}")
    print(f"Label unique values:      {result['label_unique']}")

    if not result["shape_match"]:
        print("✗ Shape mismatch, cannot compute metrics.")
        return result

    check_prediction_values(prediction, "Prediction")
    check_prediction_values(label, "Ground truth")

    result["gtvp_dsc"] = binary_dice(prediction == 1, label == 1)

    pred_n = (prediction == 2)
    gt_n = (label == 2)

    matched_inter_n, pred_sum_n, gt_sum_n = lesion_aware_aggregated_dsc_parts(pred_n, gt_n)

    result["gtvn_matched_intersection"] = matched_inter_n
    result["gtvn_pred_voxels"] = pred_sum_n
    result["gtvn_gt_voxels"] = gt_sum_n
    result["gtvn_case_binary_dsc"] = binary_dice(pred_n, gt_n)

    denom_case_proxy = pred_sum_n + gt_sum_n
    if denom_case_proxy == 0:
        result["gtvn_agg_dsc_case_proxy"] = 1.0
    else:
        result["gtvn_agg_dsc_case_proxy"] = float(2.0 * matched_inter_n / denom_case_proxy)

    tp, fp, fn = lesion_detection_counts(pred_n, gt_n, iou_threshold=GTVN_IOU_THRESHOLD)
    result["gtvn_tp"] = tp
    result["gtvn_fp"] = fp
    result["gtvn_fn"] = fn

    denom_f1 = 2 * tp + fp + fn
    if denom_f1 == 0:
        result["gtvn_f1_case"] = 1.0
    else:
        result["gtvn_f1_case"] = float(2.0 * tp / denom_f1)

    print(f"GTVp Dice: {result['gtvp_dsc']:.4f}")
    print(f"GTVn binary Dice (debug): {result['gtvn_case_binary_dsc']:.4f}")
    print(f"GTVn lesion-aware DSC proxy (case): {result['gtvn_agg_dsc_case_proxy']:.4f}")
    print(f"GTVn TP/FP/FN: {tp}/{fp}/{fn}")
    print(f"GTVn lesion F1(case): {result['gtvn_f1_case']:.4f}")

    return result


def get_case_name(pet_path):
    filename = os.path.basename(pet_path)
    if filename.endswith("_pet.npz"):
        return filename[:-8]
    return os.path.splitext(filename)[0]


def prepare_output_dirs(project_root, config):
    base_dir = os.path.join(
        project_root,
        "outputs",
        config.experiment_name,
        f"fold_{config.fold}",
    )

    prediction_dir = os.path.join(base_dir, "predictions_npz")
    submission_dir = os.path.join(base_dir, "submission_nifti")
    metrics_dir = os.path.join(base_dir, "metrics")
    vis_dir = os.path.join(base_dir, "visualizations")

    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    return prediction_dir, submission_dir, metrics_dir, vis_dir


def save_prediction_npz(prediction, prediction_dir, case_name):
    save_path = os.path.join(prediction_dir, f"{case_name}_pred.npz")
    np.savez_compressed(save_path, prediction=prediction)
    print(f"✓ Internal prediction NPZ saved to: {save_path}")
    return save_path


def numpy_xyz_to_sitk(arr_xyz: np.ndarray, meta: Dict) -> sitk.Image:
    """
    Convert numpy array [X, Y, Z] to SimpleITK image with correct meta.
    sitk expects [Z, Y, X].
    """
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
    img = sitk.GetImageFromArray(arr_zyx)

    if meta is None:
        raise ValueError("Meta is required to convert prediction from resampled space to SITK.")

    img.SetSpacing(tuple(float(x) for x in meta["spacing"]))
    img.SetOrigin(tuple(float(x) for x in meta["origin"]))
    img.SetDirection(tuple(float(x) for x in meta["direction"]))
    return img


def sitk_to_numpy_xyz(img: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image to numpy [X, Y, Z]."""
    arr_zyx = sitk.GetArrayFromImage(img)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    return arr_xyz


def extract_pre_crop_and_crop_meta(pet_meta: Dict) -> Tuple[Dict, Dict, Optional[List[int]]]:
    """
    Parse metadata from PET-only preprocess format.

    Returns:
        pre_crop_meta: full resampled PET grid meta
        cropped_meta: cropped PET grid meta
        crop_bbox_xyz: [x0, x1, y0, y1, z0, z1] or None
    """
    if pet_meta is None:
        raise ValueError("PET meta is missing from NPZ.")

    if "pre_crop_resampled_meta" in pet_meta and "cropped_resampled_meta" in pet_meta:
        pre_crop_meta = pet_meta["pre_crop_resampled_meta"]
        cropped_meta = pet_meta["cropped_resampled_meta"]
        crop_bbox_xyz = pet_meta.get("crop_bbox_xyz", None)
        return pre_crop_meta, cropped_meta, crop_bbox_xyz

    if "resampled_meta" in pet_meta:
        meta = pet_meta["resampled_meta"]
        return meta, meta, None

    required_keys = {"spacing", "origin", "direction", "size"}
    if required_keys.issubset(set(pet_meta.keys())):
        return pet_meta, pet_meta, None

    raise ValueError(f"Cannot parse PET meta keys: {list(pet_meta.keys())}")


def uncrop_prediction_to_full_resampled(
    prediction_xyz: np.ndarray,
    pre_crop_meta: Dict,
    cropped_meta: Dict,
    crop_bbox_xyz: Optional[List[int]],
) -> np.ndarray:
    """
    Restore cropped prediction [X_crop, Y_crop, Z_crop] to full resampled PET grid [X, Y, Z].

    If crop_bbox_xyz is None, assumes prediction is already on the full resampled grid.
    """
    if crop_bbox_xyz is None:
        expected_size = tuple(int(x) for x in pre_crop_meta["size"])
        if tuple(prediction_xyz.shape) != expected_size:
            raise ValueError(
                f"No crop bbox available, but prediction shape {prediction_xyz.shape} "
                f"does not match full resampled size {expected_size}"
            )
        return prediction_xyz.astype(np.uint8)

    if len(crop_bbox_xyz) != 6:
        raise ValueError(f"crop_bbox_xyz must have length 6, got: {crop_bbox_xyz}")

    x0, x1, y0, y1, z0, z1 = [int(v) for v in crop_bbox_xyz]
    expected_crop_size = (x1 - x0, y1 - y0, z1 - z0)

    if tuple(prediction_xyz.shape) != expected_crop_size:
        raise ValueError(
            f"Prediction cropped shape mismatch: pred={prediction_xyz.shape}, "
            f"expected from crop_bbox_xyz={expected_crop_size}"
        )

    full_size = tuple(int(x) for x in pre_crop_meta["size"])  # (X, Y, Z)
    full_pred = np.zeros(full_size, dtype=np.uint8)
    full_pred[x0:x1, y0:y1, z0:z1] = prediction_xyz.astype(np.uint8)
    return full_pred


def infer_original_label_path(label_meta: Dict) -> Optional[str]:
    if label_meta is None:
        return None
    raw_path = label_meta.get("raw_path", None)
    if isinstance(raw_path, str) and os.path.exists(raw_path):
        return raw_path
    return None


def load_original_label_on_ct_grid(original_label_path: str, original_ct_img: sitk.Image) -> np.ndarray:
    gt_img = sitk.ReadImage(original_label_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_ct_img)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    gt_img = resampler.Execute(gt_img)

    gt_np = sitk_to_numpy_xyz(gt_img).astype(np.uint8)
    check_prediction_values(gt_np, "Original-space ground truth")
    return gt_np


def assert_same_grid(img_a: sitk.Image, img_b: sitk.Image, atol: float = 1e-6):
    """Strict grid check for official submission mask vs original CT."""
    if img_a.GetSize() != img_b.GetSize():
        raise ValueError(f"Grid size mismatch: {img_a.GetSize()} vs {img_b.GetSize()}")

    for name, va, vb in [
        ("spacing", img_a.GetSpacing(), img_b.GetSpacing()),
        ("origin", img_a.GetOrigin(), img_b.GetOrigin()),
        ("direction", img_a.GetDirection(), img_b.GetDirection()),
    ]:
        va = np.array(va, dtype=np.float64)
        vb = np.array(vb, dtype=np.float64)
        if not np.allclose(va, vb, atol=atol):
            raise ValueError(f"Grid {name} mismatch: {va} vs {vb}")


def save_prediction_nifti_in_original_ct_space(
    prediction_xyz: np.ndarray,
    pet_meta: Dict,
    original_ct_path: str,
    submission_dir: str,
    case_name: str,
) -> Tuple[str, sitk.Image]:
    """
    Save cropped-space prediction as NIfTI in the original CT space.

    Steps:
      1) parse crop/full-resampled PET metadata
      2) uncrop prediction back to full resampled PET grid
      3) convert full-resampled prediction to SITK image
      4) resample to original CT grid using nearest neighbor
      5) save and verify official submission format
    """
    if not os.path.exists(original_ct_path):
        raise FileNotFoundError(f"Original CT not found: {original_ct_path}")

    check_prediction_values(prediction_xyz, "Prediction before export")

    pre_crop_meta, cropped_meta, crop_bbox_xyz = extract_pre_crop_and_crop_meta(pet_meta)

    print("\n=== Export to Original CT Space ===")
    print(f"Cropped prediction shape: {prediction_xyz.shape}")
    print(f"Full resampled PET size: {tuple(pre_crop_meta['size'])}")
    print(f"Cropped PET size: {tuple(cropped_meta['size'])}")
    print(f"crop_bbox_xyz: {crop_bbox_xyz}")

    full_resampled_prediction_xyz = uncrop_prediction_to_full_resampled(
        prediction_xyz=prediction_xyz,
        pre_crop_meta=pre_crop_meta,
        cropped_meta=cropped_meta,
        crop_bbox_xyz=crop_bbox_xyz,
    )
    check_prediction_values(full_resampled_prediction_xyz, "Full-resampled prediction")

    print(f"Restored full-resampled prediction shape: {full_resampled_prediction_xyz.shape}")

    pred_full_resampled_img = numpy_xyz_to_sitk(
        full_resampled_prediction_xyz.astype(np.uint8),
        pre_crop_meta,
    )

    original_ct_img = sitk.ReadImage(original_ct_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_ct_img)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    pred_original_img = resampler.Execute(pred_full_resampled_img)
    pred_original_img = sitk.Cast(pred_original_img, sitk.sitkUInt8)

    assert_same_grid(pred_original_img, original_ct_img)

    save_path = os.path.join(submission_dir, f"{case_name}.nii.gz")
    sitk.WriteImage(pred_original_img, save_path)

    saved_img = sitk.ReadImage(save_path)
    assert_same_grid(saved_img, original_ct_img)
    saved_np = sitk_to_numpy_xyz(saved_img).astype(np.uint8)
    check_prediction_values(saved_np, "Saved submission mask")

    print(f"✓ Official submission NIfTI mask saved to: {save_path}")
    print(f"  Size: {saved_img.GetSize()}")
    print(f"  Spacing: {saved_img.GetSpacing()}")

    return save_path, saved_img


def append_metrics_csv(metrics_dir, case_name, eval_result, prediction, label, eval_space: str, min_gtvn_size: int):
    csv_path = os.path.join(metrics_dir, "metrics.csv")

    pred_unique_str = "|".join(map(str, eval_result.get("pred_unique", [])))
    label_unique_str = "|".join(map(str, eval_result.get("label_unique", []))) if label is not None else ""

    row = {
        "case_id": case_name,
        "eval_space": eval_space,
        "shape_match": eval_result.get("shape_match", False),
        "prediction_shape": str(tuple(prediction.shape)),
        "label_shape": str(tuple(label.shape)) if label is not None else "",
        "pred_unique": pred_unique_str,
        "label_unique": label_unique_str,
        "gtvp_dsc": eval_result.get("gtvp_dsc", np.nan),
        "gtvn_case_binary_dsc": eval_result.get("gtvn_case_binary_dsc", np.nan),
        "gtvn_matched_intersection": eval_result.get("gtvn_matched_intersection", 0),
        "gtvn_pred_voxels": eval_result.get("gtvn_pred_voxels", 0),
        "gtvn_gt_voxels": eval_result.get("gtvn_gt_voxels", 0),
        "gtvn_agg_dsc_case_proxy": eval_result.get("gtvn_agg_dsc_case_proxy", np.nan),
        "gtvn_tp": eval_result.get("gtvn_tp", 0),
        "gtvn_fp": eval_result.get("gtvn_fp", 0),
        "gtvn_fn": eval_result.get("gtvn_fn", 0),
        "gtvn_f1_case": eval_result.get("gtvn_f1_case", np.nan),
        "min_gtvn_size": int(min_gtvn_size),
    }

    fieldnames = list(row.keys())

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    updated = False
    for i, existing_row in enumerate(existing_rows):
        if (
            existing_row["case_id"] == case_name and
            existing_row.get("eval_space", "") == eval_space and
            int(existing_row.get("min_gtvn_size", -1)) == int(min_gtvn_size)
        ):
            existing_rows[i] = row
            updated = True
            break

    if not updated:
        existing_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"✓ Metrics saved to: {csv_path}")
    return csv_path


def update_summary_csv(metrics_dir: str):
    csv_path = os.path.join(metrics_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        return None

    summary_rows = []
    group_keys = sorted(set((row["eval_space"], row.get("min_gtvn_size", "0")) for row in rows))

    for eval_space, min_gtvn_size in group_keys:
        sub = [
            r for r in rows
            if r["eval_space"] == eval_space and str(r.get("min_gtvn_size", "0")) == str(min_gtvn_size)
        ]
        if len(sub) == 0:
            continue

        gtvp_vals = []
        gtvn_matched_inter_sum = 0
        gtvn_pred_sum = 0
        gtvn_gt_sum = 0
        gtvn_tp = 0
        gtvn_fp = 0
        gtvn_fn = 0

        for r in sub:
            gtvp = r["gtvp_dsc"]
            if gtvp not in ("", "nan", "NaN"):
                gtvp_vals.append(float(gtvp))

            gtvn_matched_inter_sum += int(float(r["gtvn_matched_intersection"]))
            gtvn_pred_sum += int(float(r["gtvn_pred_voxels"]))
            gtvn_gt_sum += int(float(r["gtvn_gt_voxels"]))
            gtvn_tp += int(float(r["gtvn_tp"]))
            gtvn_fp += int(float(r["gtvn_fp"]))
            gtvn_fn += int(float(r["gtvn_fn"]))

        gtvp_mean_dsc = float(np.mean(gtvp_vals)) if len(gtvp_vals) > 0 else np.nan

        denom_dsc = gtvn_pred_sum + gtvn_gt_sum
        gtvn_agg_dsc = 1.0 if denom_dsc == 0 else float(2.0 * gtvn_matched_inter_sum / denom_dsc)

        denom_f1 = 2 * gtvn_tp + gtvn_fp + gtvn_fn
        gtvn_f1agg = 1.0 if denom_f1 == 0 else float(2.0 * gtvn_tp / denom_f1)

        proxy_score = (
            np.nan if np.isnan(gtvp_mean_dsc)
            else float(0.5 * gtvp_mean_dsc + 0.25 * gtvn_agg_dsc + 0.25 * gtvn_f1agg)
        )

        summary_rows.append({
            "eval_space": eval_space,
            "min_gtvn_size": int(min_gtvn_size),
            "num_cases": len(sub),
            "gtvp_mean_dsc": gtvp_mean_dsc,
            "gtvn_agg_dsc": gtvn_agg_dsc,
            "gtvn_f1agg": gtvn_f1agg,
            "gtvn_tp": gtvn_tp,
            "gtvn_fp": gtvn_fp,
            "gtvn_fn": gtvn_fn,
            "task1_proxy_score": proxy_score,
        })

    summary_csv = os.path.join(metrics_dir, "metrics_summary.csv")
    fieldnames = list(summary_rows[0].keys())

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"✓ Summary metrics saved to: {summary_csv}")
    return summary_csv


def choose_slice_index(prediction, label=None):
    if label is not None and np.any(label > 0):
        z_scores = (label > 0).sum(axis=(0, 1))
        return int(np.argmax(z_scores))

    if np.any(prediction > 0):
        z_scores = (prediction > 0).sum(axis=(0, 1))
        return int(np.argmax(z_scores))

    return prediction.shape[2] // 2


def normalize_for_display(arr):
    arr = arr.astype(np.float32)
    vmin = np.percentile(arr, 1)
    vmax = np.percentile(arr, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin)
    return arr


def save_visualization_png(pet_data, prediction, vis_dir, case_name, label=None):
    slice_idx = choose_slice_index(prediction, label)

    pet_slice = pet_data[:, :, slice_idx]
    pred_slice = prediction[:, :, slice_idx]
    pet_disp = normalize_for_display(pet_slice)

    if label is not None:
        label_slice = label[:, :, slice_idx]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        label_slice = None
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(pet_disp, cmap="gray")
    axes[0].set_title(f"PET (z={slice_idx})")
    axes[0].axis("off")

    axes[1].imshow(pet_disp, cmap="gray")
    pred1 = (pred_slice == 1).astype(np.float32)
    pred2 = (pred_slice == 2).astype(np.float32)

    if pred1.max() > 0:
        axes[1].imshow(np.ma.masked_where(pred1 == 0, pred1), alpha=0.5, cmap="Reds")
    if pred2.max() > 0:
        axes[1].imshow(np.ma.masked_where(pred2 == 0, pred2), alpha=0.5, cmap="Blues")

    axes[1].set_title("PET + Prediction")
    axes[1].axis("off")

    if label is not None:
        axes[2].imshow(pet_disp, cmap="gray")
        gt1 = (label_slice == 1).astype(np.float32)
        gt2 = (label_slice == 2).astype(np.float32)

        if gt1.max() > 0:
            axes[2].imshow(np.ma.masked_where(gt1 == 0, gt1), alpha=0.5, cmap="Reds")
        if gt2.max() > 0:
            axes[2].imshow(np.ma.masked_where(gt2 == 0, gt2), alpha=0.5, cmap="Blues")

        axes[2].set_title("PET + Ground Truth")
        axes[2].axis("off")

    plt.tight_layout()

    save_path = os.path.join(vis_dir, f"{case_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Visualization saved to: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Run HECKTOR Task 1 inference (PET-only)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--pet_path", type=str, required=True, help="Path to PET NPZ")
    parser.add_argument("--label_path", type=str, default=None,
                        help="Optional label NPZ path (cropped/resampled-space evaluation)")
    parser.add_argument("--original_ct_path", type=str, default=None,
                        help="Optional original raw CT NIfTI path for official submission export")
    parser.add_argument("--original_label_path", type=str, default=None,
                        help="Optional original raw label NIfTI path for original-space evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--no_sliding_window", action="store_true", help="Disable sliding window inference")
    parser.add_argument(
        "--min-gtvn-size",
        type=int,
        default=10,
        help="Remove predicted GTVn connected components smaller than this many voxels",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    model, config = load_model_from_checkpoint(args.model_path, args.device)
    if model is None or config is None:
        print("Failed to load model or config")
        return

    try:
        input_tensor, pet_data, pet_meta = load_npz_data(args.pet_path)
        print(f"✓ Data loaded successfully: {input_tensor.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    use_sliding_window = not args.no_sliding_window
    raw_prediction = run_inference(model, input_tensor, config, args.device, use_sliding_window)

    if raw_prediction is None:
        print("\n✗ Inference failed")
        return

    prediction = postprocess_prediction_task1(
        raw_prediction,
        min_gtvn_size=args.min_gtvn_size,
    )

    print("\n✓ Inference completed successfully!")
    print(f"Final postprocessed prediction shape: {prediction.shape}")
    print(f"Final postprocessed unique values: {np.unique(prediction).tolist()}")

    case_name = get_case_name(args.pet_path)
    prediction_dir, submission_dir, metrics_dir, vis_dir = prepare_output_dirs(current_dir, config)

    save_prediction_npz(prediction, prediction_dir, case_name)

    pred_original_img = None
    pred_original_np = None

    if args.original_ct_path is not None:
        try:
            _, pred_original_img = save_prediction_nifti_in_original_ct_space(
                prediction_xyz=prediction,
                pet_meta=pet_meta,
                original_ct_path=args.original_ct_path,
                submission_dir=submission_dir,
                case_name=case_name,
            )
            pred_original_np = sitk_to_numpy_xyz(pred_original_img).astype(np.uint8)
            check_prediction_values(pred_original_np, "Original-space prediction")
        except Exception as e:
            print(f"Error saving original-space NIfTI: {e}")
    else:
        print("No original CT path provided. Official submission NIfTI export was skipped.")

    eval_result = None
    eval_space = "none"
    label_for_vis = None
    label_for_csv = None
    pred_for_csv = prediction

    # Prefer original-space evaluation if original CT and original label are both available
    if args.original_label_path is not None and pred_original_np is not None and args.original_ct_path is not None:
        try:
            original_ct_img = sitk.ReadImage(args.original_ct_path)
            gt_original_np = load_original_label_on_ct_grid(args.original_label_path, original_ct_img)
            eval_result = evaluate_single_case(pred_original_np, gt_original_np)
            eval_space = "original_ct_space"
            label_for_csv = gt_original_np
            pred_for_csv = pred_original_np
        except Exception as e:
            print(f"Error evaluating in original CT space: {e}")

    # Fallback: cropped/resampled-space evaluation
    if eval_result is None and args.label_path is not None:
        try:
            label_resampled, label_meta = load_label_data(args.label_path)
            label_for_vis = label_resampled
            label_for_csv = label_resampled
            eval_result = evaluate_single_case(prediction, label_resampled)
            eval_space = "resampled_space"
            pred_for_csv = prediction

            if args.original_label_path is None:
                inferred_label_path = infer_original_label_path(label_meta)
                if inferred_label_path is not None:
                    print(f"Inferred original label path: {inferred_label_path}")
        except Exception as e:
            print(f"Error loading/evaluating resampled label: {e}")

    if label_for_vis is None and args.label_path is not None:
        try:
            label_for_vis, _ = load_label_data(args.label_path)
        except Exception:
            label_for_vis = None

    if eval_result is None:
        eval_result = {
            "shape_match": False,
            "pred_unique": np.unique(prediction).tolist(),
            "label_unique": [],
            "gtvp_dsc": np.nan,
            "gtvn_case_binary_dsc": np.nan,
            "gtvn_matched_intersection": 0,
            "gtvn_pred_voxels": int((prediction == 2).sum()),
            "gtvn_gt_voxels": 0,
            "gtvn_agg_dsc_case_proxy": np.nan,
            "gtvn_tp": 0,
            "gtvn_fp": 0,
            "gtvn_fn": 0,
            "gtvn_f1_case": np.nan,
        }
        eval_space = "none"
        label_for_csv = None
        pred_for_csv = prediction

    append_metrics_csv(
        metrics_dir=metrics_dir,
        case_name=case_name,
        eval_result=eval_result,
        prediction=pred_for_csv,
        label=label_for_csv,
        eval_space=eval_space,
        min_gtvn_size=args.min_gtvn_size,
    )

    update_summary_csv(metrics_dir)
    save_visualization_png(pet_data, prediction, vis_dir, case_name, label_for_vis)

    print("\nAll outputs saved successfully.")
    if pred_original_img is not None:
        print("Official submission file is the .nii.gz in submission_nifti/.")


if __name__ == "__main__":
    main()