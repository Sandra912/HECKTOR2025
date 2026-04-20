#!/usr/bin/env python3
"""
HECKTOR 2025 Task 1 submission inference script (PET-only, simplified).

Official requirements handled by this script:
1) Final exported mask is saved on ORIGINAL CT grid
2) Final labels are strictly {0, 1, 2}
3) One final NIfTI per case for submission
4) Supports single-model inference and multi-model ensemble inference

=======================================================================
How to use
=======================================================================

[1] Single-model, single-case inference
python scripts/inference.py \
  --model_paths outputs/unet3d/fold_0/checkpoints/best_model.pth \
  --pet_path /data/HECKTOR2025/Task1/imagesTr_resampled_npy/CHUS-027_pet.npz \
  --original_ct_path /data/HECKTOR2025/Task1/CHUS-027/CHUS-027__CT.nii.gz \
  --device cuda

[2] 5-fold ensemble, single-case inference
python scripts/inference.py \
  --model_paths \
    outputs/unet3d/fold_0/checkpoints/best_model.pth \
    outputs/unet3d/fold_1/checkpoints/best_model.pth \
    outputs/unet3d/fold_2/checkpoints/best_model.pth \
    outputs/unet3d/fold_3/checkpoints/best_model.pth \
    outputs/unet3d/fold_4/checkpoints/best_model.pth \
  --pet_path /data/HECKTOR2025/Task1/imagesTr_resampled_npy/CHUS-027_pet.npz \
  --original_ct_path /data/HECKTOR2025/Task1/CHUS-027/CHUS-027__CT.nii.gz \
  --device cuda

[3] Single-model batch inference
python scripts/inference.py \
  --model_paths outputs/unet3d/fold_0/checkpoints/best_model.pth \
  --splits_file config/splits_available.json \
  --fold 0 \
  --split_mode val \
  --data_root /data/HECKTOR2025/Task1 \
  --pet_dir /data/HECKTOR2025/Task1/imagesTr_resampled_npy \
  --device cuda

[4] 5-fold ensemble batch inference
python scripts/inference.py \
  --model_paths \
    outputs/unet3d/fold_0/checkpoints/best_model.pth \
    outputs/unet3d/fold_1/checkpoints/best_model.pth \
    outputs/unet3d/fold_2/checkpoints/best_model.pth \
    outputs/unet3d/fold_3/checkpoints/best_model.pth \
    outputs/unet3d/fold_4/checkpoints/best_model.pth \
  --splits_file config/splits_available.json \
  --fold 0 \
  --split_mode val \
  --data_root /data/HECKTOR2025/Task1 \
  --pet_dir /data/HECKTOR2025/Task1/imagesTr_resampled_npy \
  --device cuda

Notes
-----
- Pass 1 checkpoint in --model_paths for single-model inference.
- Pass multiple checkpoints in --model_paths for ensemble inference.
- Ensemble is done by averaging per-class probabilities (softmax), then argmax.
- Final submission masks are always exported on ORIGINAL CT grid.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label
from monai.inferers import sliding_window_inference

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from models import UNet3DModel, SegResNetModel, UNETRModel, SwinUNETRModel
    from config import UNet3DConfig, SegResNetConfig, UNETRConfig, SwinUNETRConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the Task1 directory")
    sys.exit(1)

VALID_LABELS = {0, 1, 2}
PET_CLIP_MIN = 0.0
PET_CLIP_MAX = 30.0


# =========================
# IO helpers
# =========================
def load_npz_payload(npz_path: str) -> Dict:
    loaded = np.load(npz_path, allow_pickle=True)
    payload = {}
    for k in loaded.files:
        value = loaded[k]
        if k == "meta" and isinstance(value, np.ndarray) and value.shape == ():
            payload[k] = value.item()
        else:
            payload[k] = value
    return payload


def get_case_name(pet_path: str) -> str:
    filename = os.path.basename(pet_path)
    if filename.endswith("_pet.npz"):
        return filename[:-8]
    return os.path.splitext(filename)[0]


def prepare_output_dirs(project_root: str, config, is_ensemble: bool = False):
    if is_ensemble:
        base_dir = os.path.join(project_root, "outputs", config.experiment_name, "ensemble")
    else:
        base_dir = os.path.join(project_root, "outputs", config.experiment_name, f"fold_{config.fold}")

    submission_dir = os.path.join(base_dir, "submission_nifti")
    os.makedirs(submission_dir, exist_ok=True)
    return submission_dir


def check_prediction_values(arr: np.ndarray, name: str = "Prediction"):
    uniq = set(np.unique(arr).tolist())
    if not uniq.issubset(VALID_LABELS):
        raise ValueError(
            f"{name} has invalid labels: {sorted(uniq)}. Expected subset of {sorted(VALID_LABELS)}."
        )


# =========================
# Postprocess
# =========================
def connected_components_3d(mask: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    mask = mask.astype(np.uint8)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num = cc_label(mask, structure=structure)

    components, sizes = [], []
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        components.append(comp)
        sizes.append(int(comp.sum()))
    return components, sizes


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask.astype(bool)

    components, sizes = connected_components_3d(mask)
    out = np.zeros_like(mask, dtype=bool)
    for comp, size in zip(components, sizes):
        if size >= min_size:
            out |= comp
    return out


def postprocess_prediction_task1(pred: np.ndarray, min_gtvn_size: int = 10) -> np.ndarray:
    """
    Keep:
    - label 1 (GTVp)
    - label 2 (GTVn), removing tiny GTVn connected components
    Enforce final label space {0,1,2}
    """
    check_prediction_values(pred, "Prediction before postprocess")

    out = np.zeros_like(pred, dtype=np.uint8)
    out[pred == 1] = 1

    pred_n = (pred == 2)
    pred_n = remove_small_components(pred_n, min_gtvn_size)

    # GTVp priority over GTVn
    pred_n = np.logical_and(pred_n, out != 1)
    out[pred_n] = 2

    check_prediction_values(out, "Prediction after postprocess")
    return out


# =========================
# SITK helpers
# =========================
def numpy_xyz_to_sitk(arr_xyz: np.ndarray, meta: Dict) -> sitk.Image:
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing(tuple(float(x) for x in meta["spacing"]))
    img.SetOrigin(tuple(float(x) for x in meta["origin"]))
    img.SetDirection(tuple(float(x) for x in meta["direction"]))
    return img


def sitk_to_numpy_xyz(img: sitk.Image) -> np.ndarray:
    return np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))


def assert_same_grid(img_a: sitk.Image, img_b: sitk.Image, atol: float = 1e-6):
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


# =========================
# Crop / restore helpers
# =========================
def extract_pre_crop_and_crop_meta(pet_meta: Dict) -> Tuple[Dict, Dict, Optional[List[int]]]:
    if pet_meta is None:
        raise ValueError("PET meta is missing from NPZ.")

    if "pre_crop_resampled_meta" in pet_meta and "cropped_resampled_meta" in pet_meta:
        return (
            pet_meta["pre_crop_resampled_meta"],
            pet_meta["cropped_resampled_meta"],
            pet_meta.get("crop_bbox_xyz", None),
        )

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
    crop_bbox_xyz: Optional[List[int]],
) -> np.ndarray:
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

    x0, x1, y0, y1, z0, z1 = map(int, crop_bbox_xyz)
    expected_crop_size = (x1 - x0, y1 - y0, z1 - z0)
    if tuple(prediction_xyz.shape) != expected_crop_size:
        raise ValueError(
            f"Prediction cropped shape mismatch: pred={prediction_xyz.shape}, "
            f"expected from crop_bbox_xyz={expected_crop_size}"
        )

    full_size = tuple(int(x) for x in pre_crop_meta["size"])
    full_pred = np.zeros(full_size, dtype=np.uint8)
    full_pred[x0:x1, y0:y1, z0:z1] = prediction_xyz.astype(np.uint8)
    return full_pred


def save_prediction_nifti_in_original_ct_space(
    prediction_xyz: np.ndarray,
    pet_meta: Dict,
    original_ct_path: str,
    submission_dir: str,
    case_name: str,
) -> str:
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
        crop_bbox_xyz=crop_bbox_xyz,
    )
    check_prediction_values(full_resampled_prediction_xyz, "Full-resampled prediction")

    pred_full_resampled_img = numpy_xyz_to_sitk(full_resampled_prediction_xyz.astype(np.uint8), pre_crop_meta)
    original_ct_img = sitk.ReadImage(original_ct_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_ct_img)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    pred_original_img = sitk.Cast(resampler.Execute(pred_full_resampled_img), sitk.sitkUInt8)

    assert_same_grid(pred_original_img, original_ct_img)

    save_path = os.path.join(submission_dir, f"{case_name}.nii.gz")
    sitk.WriteImage(pred_original_img, save_path)

    # Final safety check after save
    saved_img = sitk.ReadImage(save_path)
    assert_same_grid(saved_img, original_ct_img)
    saved_np = sitk_to_numpy_xyz(saved_img).astype(np.uint8)
    check_prediction_values(saved_np, "Saved submission mask")

    print(f"✓ Official submission NIfTI saved to: {save_path}")
    print(f"  Size: {saved_img.GetSize()}")
    print(f"  Spacing: {saved_img.GetSpacing()}")

    return save_path


# =========================
# Model loading
# =========================
def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_config" not in checkpoint:
        raise ValueError("No model_config found in checkpoint")

    model_config = checkpoint["model_config"]
    experiment_name = model_config.get("experiment_name", "unet3d")
    print(f"Found model config for experiment: {experiment_name}")

    if experiment_name == "unet3d":
        config = UNet3DConfig()
        model_cls = UNet3DModel
    elif experiment_name == "segresnet":
        config = SegResNetConfig()
        model_cls = SegResNetModel
    elif experiment_name == "unetr":
        config = UNETRConfig()
        model_cls = UNETRModel
    elif experiment_name == "swinunetr":
        config = SwinUNETRConfig()
        model_cls = SwinUNETRModel
    else:
        raise ValueError(f"Unknown experiment type: {experiment_name}")

    for key, value in model_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.input_channels = 1

    model = model_cls(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully | fold={config.fold} | spatial_size={config.spatial_size}")
    return model, config


def load_models_from_checkpoints(checkpoint_paths: List[str], device: str = "cuda"):
    models = []
    configs = []

    for ckpt_path in checkpoint_paths:
        model, config = load_model_from_checkpoint(ckpt_path, device)
        models.append(model)
        configs.append(config)

    ref_config = configs[0]
    for i, cfg in enumerate(configs[1:], start=1):
        if tuple(cfg.spatial_size) != tuple(ref_config.spatial_size):
            raise ValueError(
                f"Model {i} spatial_size mismatch: {cfg.spatial_size} vs {ref_config.spatial_size}"
            )
        if int(cfg.num_classes) != int(ref_config.num_classes):
            raise ValueError(
                f"Model {i} num_classes mismatch: {cfg.num_classes} vs {ref_config.num_classes}"
            )

    print(f"✓ Loaded {len(models)} model(s) for inference")
    return models, ref_config


# =========================
# Data loading
# =========================
def preprocess_pet_for_model(pet_data_xyz: np.ndarray) -> np.ndarray:
    pet = pet_data_xyz.astype(np.float32)
    pet = np.clip(pet, PET_CLIP_MIN, PET_CLIP_MAX)
    pet = pet / PET_CLIP_MAX
    return pet


def load_npz_data(pet_path: str):
    print(f"Loading PET data from: {pet_path}")
    pet_payload = load_npz_payload(pet_path)
    pet_data = pet_payload["image"]
    pet_meta = pet_payload.get("meta", None)

    if pet_data.ndim != 4:
        raise ValueError(f"Expected PET shape [1, X, Y, Z], got PET={pet_data.shape}")

    if pet_data.shape[0] == 1:
        pet_data = pet_data.squeeze(0)

    pet_model = preprocess_pet_for_model(pet_data)
    tensor = torch.from_numpy(np.expand_dims(pet_model, axis=0)).float().unsqueeze(0)
    return tensor, pet_meta


# =========================
# Inference
# =========================
def run_inference(model, input_tensor, config, device="cuda", use_sliding_window=False):
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        if use_sliding_window:
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
            output = model(input_tensor)

    if output.shape[1] > 1:
        prediction = torch.softmax(output, dim=1).argmax(dim=1)
    else:
        prediction = (torch.sigmoid(output) > 0.5).long()

    prediction_np = prediction.cpu().numpy().squeeze(0).astype(np.uint8)
    check_prediction_values(prediction_np, "Raw prediction")
    return prediction_np


def run_ensemble_inference(models, input_tensor, config, device="cuda", use_sliding_window=False):
    input_tensor = input_tensor.to(device)
    prob_sum = None

    with torch.no_grad():
        for model_idx, model in enumerate(models, start=1):
            print(f"Running model {model_idx}/{len(models)}")

            if use_sliding_window:
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
                output = model(input_tensor)

            if output.shape[1] > 1:
                probs = torch.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            prob_sum = probs if prob_sum is None else prob_sum + probs

    avg_probs = prob_sum / len(models)

    if avg_probs.shape[1] > 1:
        prediction = torch.argmax(avg_probs, dim=1)
    else:
        prediction = (avg_probs > 0.5).long()

    prediction_np = prediction.cpu().numpy().squeeze(0).astype(np.uint8)
    check_prediction_values(prediction_np, "Raw ensemble prediction")
    return prediction_np


# =========================
# Single / batch drivers
# =========================
def load_case_ids_from_splits(splits_file: str, fold_idx: int, split_mode: str = "val") -> List[str]:
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError("Splits file must contain a non-empty list.")
    if not (0 <= fold_idx < len(splits)):
        raise ValueError(f"fold_idx={fold_idx} out of range. Found {len(splits)} folds.")

    fold = splits[fold_idx]
    if split_mode == "train":
        return sorted(fold["train"])
    if split_mode == "val":
        return sorted(fold["val"])
    if split_mode == "both":
        return sorted(set(fold["train"]) | set(fold["val"]))
    raise ValueError("split_mode must be one of: train, val, both")


def build_case_paths(case_id: str, data_root: str, pet_dir: str) -> Dict[str, str]:
    return {
        "pet_path": os.path.join(pet_dir, f"{case_id}_pet.npz"),
        "original_ct_path": os.path.join(data_root, case_id, f"{case_id}__CT.nii.gz"),
    }


def check_case_files(paths: Dict[str, str]) -> List[str]:
    missing = []
    for name, path in paths.items():
        if path is not None and not os.path.exists(path):
            missing.append(f"{name}: {path}")
    return missing


def run_single_case(
    models,
    config,
    pet_path: str,
    original_ct_path: str,
    device: str,
    min_gtvn_size: int,
    use_sliding_window: bool,
):
    input_tensor, pet_meta = load_npz_data(pet_path)

    if len(models) == 1:
        raw_prediction = run_inference(models[0], input_tensor, config, device, use_sliding_window)
    else:
        raw_prediction = run_ensemble_inference(models, input_tensor, config, device, use_sliding_window)

    prediction = postprocess_prediction_task1(raw_prediction, min_gtvn_size=min_gtvn_size)
    
    # ===== DEBUG: check overlap between GTVp and GTVn =====
    overlap = np.logical_and(prediction == 1, prediction == 2)
    print("Overlap voxels:", overlap.sum())

    case_name = get_case_name(pet_path)
    is_ensemble = len(models) > 1
    submission_dir = prepare_output_dirs(PROJECT_ROOT, config, is_ensemble=is_ensemble)

    save_path = save_prediction_nifti_in_original_ct_space(
        prediction_xyz=prediction,
        pet_meta=pet_meta,
        original_ct_path=original_ct_path,
        submission_dir=submission_dir,
        case_name=case_name,
    )
    return save_path


def run_batch_cases(args):
    case_ids = load_case_ids_from_splits(args.splits_file, args.fold, args.split_mode)
    if args.limit is not None:
        case_ids = case_ids[:args.limit]

    print(f"Splits file: {args.splits_file}")
    print(f"Fold: {args.fold}")
    print(f"Split mode: {args.split_mode}")
    print(f"Models: {args.model_paths}")
    print(f"Data root: {args.data_root}")
    print(f"PET dir: {args.pet_dir}")
    print(f"Device: {args.device}")
    print(f"min_gtvn_size: {args.min_gtvn_size}")
    print(f"Number of cases: {len(case_ids)}")

    models, config = load_models_from_checkpoints(args.model_paths, args.device)
    use_sliding_window = not args.no_sliding_window

    start_all = time.time()
    total_runs = success_runs = failed_runs = 0
    skipped_cases = []

    for idx, case_id in enumerate(case_ids, start=1):
        print(f"\n[{idx}/{len(case_ids)}] {case_id}")

        paths = build_case_paths(case_id, args.data_root, args.pet_dir)
        missing = check_case_files(paths)
        if missing:
            print(f"[Skip] Missing files for {case_id}")
            for item in missing:
                print("   ", item)
            skipped_cases.append((case_id, missing))
            continue

        total_runs += 1
        try:
            run_single_case(
                models=models,
                config=config,
                pet_path=paths["pet_path"],
                original_ct_path=paths["original_ct_path"],
                device=args.device,
                min_gtvn_size=args.min_gtvn_size,
                use_sliding_window=use_sliding_window,
            )
            success_runs += 1
        except Exception as e:
            failed_runs += 1
            print(f"[Failed] {case_id}")
            print(e)

    elapsed = time.time() - start_all
    print("\n" + "=" * 100)
    print("Batch inference completed.")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")
    print(f"Total attempted runs: {total_runs}")
    print(f"Successful runs: {success_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Skipped cases: {len(skipped_cases)}")


# =========================
# Args / entry
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Run HECKTOR Task 1 submission inference")

    parser.add_argument(
        "--model_paths",
        nargs="+",
        type=str,
        required=True,
        help="One or more model checkpoint paths (.pth)."
    )

    # single-case mode
    parser.add_argument("--pet_path", type=str, default=None, help="Path to PET NPZ")
    parser.add_argument("--original_ct_path", type=str, default=None, help="Path to original raw CT NIfTI")
    parser.add_argument("--min-gtvn-size", type=int, default=10, help="Remove tiny GTVn components")

    # batch mode
    parser.add_argument("--splits_file", type=str, default=None, help="Path to splits JSON")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--split_mode", type=str, default="val", choices=["train", "val", "both"])
    parser.add_argument("--data_root", type=str, default=None, help="Root folder containing raw case folders")
    parser.add_argument("--pet_dir", type=str, default=None, help="Directory containing *_pet.npz")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N cases")

    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--no_sliding_window", action="store_true", help="Disable sliding window inference")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    print(f"Using device: {args.device}")

    if args.splits_file is not None:
        if args.data_root is None or args.pet_dir is None:
            raise ValueError("Batch mode requires --data_root and --pet_dir")
        run_batch_cases(args)
        return

    if args.pet_path is None:
        raise ValueError("Single-case mode requires --pet_path")
    if args.original_ct_path is None:
        raise ValueError("Single-case mode requires --original_ct_path")

    models, config = load_models_from_checkpoints(args.model_paths, args.device)
    use_sliding_window = not args.no_sliding_window

    run_single_case(
        models=models,
        config=config,
        pet_path=args.pet_path,
        original_ct_path=args.original_ct_path,
        device=args.device,
        min_gtvn_size=args.min_gtvn_size,
        use_sliding_window=use_sliding_window,
    )


if __name__ == "__main__":
    main()