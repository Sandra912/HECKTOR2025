#!/usr/bin/env python3
"""
Training script for HECKTOR 2025 Task 1 segmentation models (PET-only version).

Examples
--------
Single fold training:
python scripts/train.py --config unet3d --fold 0

Run standard 5-fold training:
python scripts/train.py --config unet3d --run-all-folds

Single fold Optuna:
python scripts/train.py \
  --config unet3d \
  --fold 0 \
  --optuna \
  --n-trials 40 \
  --epochs 10 \
  --validate-every 2 \
  --optuna-metric gtvn_f1agg \
  --optuna-losses dice_ce dice_focal \
  --study-name hecktor_task1_unet3d_f0 \
  --storage sqlite:///optuna_unet3d_f0.db \
  2>&1 | tee optuna_logs/optuna_unet3d_f0.log

TensorBoard:
tensorboard --logdir ./logs --port 6006

Goals:
- gtvn_f1agg ↑ (most important)
- gtvn_fp ↓
- gtvn_fn should not explode
"""

import os
import sys
import json
import csv
import copy
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import label as cc_label
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UNet3DConfig, SegResNetConfig, UNETRConfig, SwinUNETRConfig
from models import UNet3DModel, SegResNetModel, UNETRModel, SwinUNETRModel
from data import get_dataloaders
from utils.losses import get_loss_function
from utils.logging import setup_logging


VALID_LABELS = {0, 1, 2}
GTVN_IOU_THRESHOLD = 0.30
NUM_FOLDS = 5


def binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice for one binary mask. If both empty, return 1.0."""
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
    HECKTOR-style lesion-aware aggregated DSC parts for GTVn.
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


def check_values(arr: np.ndarray, name: str):
    uniq = set(np.unique(arr).tolist())
    if not uniq.issubset(VALID_LABELS):
        raise ValueError(
            f"{name} has invalid labels: {sorted(uniq)}, expected subset of {sorted(VALID_LABELS)}"
        )


def postprocess_prediction_task1(
    pred: np.ndarray,
    min_gtvn_size: int = 30,
) -> np.ndarray:
    """
    Task 1 postprocessing for validation selection:
    - keep GTVp as-is
    - remove tiny GTVn connected components
    - enforce label space {0,1,2}
    """
    check_values(pred, "Prediction before postprocess")

    out = np.zeros_like(pred, dtype=np.uint8)

    out[pred == 1] = 1

    pred_n = (pred == 2)
    pred_n = remove_small_components(pred_n, min_gtvn_size)

    pred_n = np.logical_and(pred_n, out != 1)
    out[pred_n] = 2

    check_values(out, "Prediction after postprocess")
    return out


def update_task1_accumulators(acc: Dict, pred: np.ndarray, gt: np.ndarray):
    """Update Task 1 metrics accumulators from one case."""
    check_values(pred, "Prediction")
    check_values(gt, "Ground truth")

    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shape mismatch: pred={pred.shape}, gt={gt.shape}")

    gtvp_dsc = binary_dice(pred == 1, gt == 1)
    acc["gtvp_case_dscs"].append(gtvp_dsc)

    pred_n = (pred == 2)
    gt_n = (gt == 2)

    matched_inter_n, pred_sum_n, gt_sum_n = lesion_aware_aggregated_dsc_parts(pred_n, gt_n)
    acc["gtvn_matched_intersection_sum"] += matched_inter_n
    acc["gtvn_pred_sum"] += pred_sum_n
    acc["gtvn_gt_sum"] += gt_sum_n

    tp, fp, fn = lesion_detection_counts(pred_n, gt_n, iou_threshold=GTVN_IOU_THRESHOLD)
    acc["gtvn_tp"] += tp
    acc["gtvn_fp"] += fp
    acc["gtvn_fn"] += fn

    acc["num_cases"] += 1


def finalize_task1_metrics(acc: Dict) -> Dict[str, float]:
    """Compute final Task 1-style validation metrics."""
    if len(acc["gtvp_case_dscs"]) == 0:
        gtvp_mean_dsc = 0.0
    else:
        gtvp_mean_dsc = float(np.mean(acc["gtvp_case_dscs"]))

    denom_dsc = acc["gtvn_pred_sum"] + acc["gtvn_gt_sum"]
    if denom_dsc == 0:
        gtvn_agg_dsc = 1.0
    else:
        gtvn_agg_dsc = float(2.0 * acc["gtvn_matched_intersection_sum"] / denom_dsc)

    denom_f1 = 2 * acc["gtvn_tp"] + acc["gtvn_fp"] + acc["gtvn_fn"]
    if denom_f1 == 0:
        gtvn_f1agg = 1.0
    else:
        gtvn_f1agg = float(2.0 * acc["gtvn_tp"] / denom_f1)

    task1_proxy_score = float(
        0.5 * gtvp_mean_dsc +
        0.25 * gtvn_agg_dsc +
        0.25 * gtvn_f1agg
    )

    return {
        "gtvp_mean_dsc": gtvp_mean_dsc,
        "gtvn_agg_dsc": gtvn_agg_dsc,
        "gtvn_f1agg": gtvn_f1agg,
        "task1_proxy_score": task1_proxy_score,
        "gtvn_tp": int(acc["gtvn_tp"]),
        "gtvn_fp": int(acc["gtvn_fp"]),
        "gtvn_fn": int(acc["gtvn_fn"]),
        "num_cases": int(acc["num_cases"]),
    }


def evaluate_epoch(
    model,
    loader,
    criterion,
    device,
    config,
    use_sliding_window: bool = True,
    min_gtvn_size: int = 30,
    sw_overlap: float = 0.41,
):
    """
    Evaluate one epoch with Task 1 aligned metrics:
      - GTVp mean DSC
      - GTVn lesion-aware aggregated DSC
      - GTVn aggregated detection F1 (IoU > 0.3)
    Includes validation-time GTVn postprocessing.
    """
    model.eval()
    total_loss = 0.0

    roi_size = config.spatial_size
    sw_batch_size = 1

    acc = {
        "gtvp_case_dscs": [],
        "gtvn_matched_intersection_sum": 0,
        "gtvn_pred_sum": 0,
        "gtvn_gt_sum": 0,
        "gtvn_tp": 0,
        "gtvn_fp": 0,
        "gtvn_fn": 0,
        "num_cases": 0,
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float().to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)

            if use_sliding_window:
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=sw_overlap,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0,
                    sw_device=device,
                    device=device,
                )
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)

            if labels.ndim == 5 and labels.shape[1] == 1:
                labels_eval = labels.squeeze(1)
            else:
                labels_eval = labels

            pred_np = pred.detach().cpu().numpy()
            label_np = labels_eval.detach().cpu().numpy()

            for i in range(pred_np.shape[0]):
                pred_case = pred_np[i].astype(np.uint8)
                gt_case = label_np[i].astype(np.uint8)

                pred_case = postprocess_prediction_task1(
                    pred_case,
                    min_gtvn_size=min_gtvn_size,
                )

                update_task1_accumulators(acc, pred_case, gt_case)

            del images, labels, outputs, pred, labels_eval, pred_np, label_np, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, len(loader))
    metrics = finalize_task1_metrics(acc)
    metrics["val_loss"] = float(avg_loss)
    metrics["val_min_gtvn_size"] = int(min_gtvn_size)
    metrics["sw_overlap"] = float(sw_overlap)
    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch (PET-only)."""
    model.train()
    total_loss = 0.0

    for bi, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = batch["image"].float().to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader))
    return avg_loss




def parse_args():
    parser = argparse.ArgumentParser(description="Train HECKTOR Task 1 segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="unet3d",
        choices=["unet3d", "segresnet", "unetr", "swinunetr"],
        help="Model configuration to use",
    )
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold to use")
    parser.add_argument(
        "--run-all-folds",
        action="store_true",
        help="Run all 5 folds sequentially",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, help="Override device from config (e.g., 'cpu', 'cuda')")
    parser.add_argument("--epochs", type=int, default=350, help="Override number of epochs")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--validate-every", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument(
        "--min-gtvn-size",
        type=int,
        default=30,
        help="Remove predicted GTVn connected components smaller than this many voxels during validation",
    )

    # Optuna
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, default="hecktor_task1")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage, e.g. sqlite:///optuna.db",
    )
    parser.add_argument(
        "--optuna-metric",
        type=str,
        default="gtvn_f1agg",
        choices=["gtvn_f1agg", "task1_proxy_score", "gtvn_agg_dsc", "gtvp_mean_dsc"],
        help="Metric to maximize in Optuna",
    )
    parser.add_argument(
        "--optuna-losses",
        nargs="+",
        default=["dice_focal"],
        help="Loss names available to Optuna. Use only losses supported by utils.losses.get_loss_function.",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for Optuna study.optimize()",
    )
    parser.add_argument(
        "--disable-tensorboard-in-optuna",
        action="store_true",
        help="Disable TensorBoard writers during Optuna trials to reduce I/O",
    )
    # 调优
    parser.add_argument(
        "--optuna-folds",
        type=str,
        default="0,2,4",
        help="Comma-separated folds used during Optuna, e.g. '0,1,2' or '0,1,2,3,4'",
    )
    parser.add_argument(
        "--best-model-metric",
        type=str,
        default="gtvn_f1agg",
        choices=["gtvn_f1agg", "task1_proxy_score", "gtvn_agg_dsc", "gtvp_mean_dsc"],
        help="Metric used to select best checkpoint in normal training",
    )

    return parser.parse_args()

def parse_fold_list(folds_str: str) -> List[int]:
    folds = [int(x.strip()) for x in folds_str.split(",") if x.strip() != ""]
    if len(folds) == 0:
        raise ValueError("--optuna-folds must contain at least one fold.")
    bad = [f for f in folds if f < 0 or f >= NUM_FOLDS]
    if len(bad) > 0:
        raise ValueError(f"Invalid folds in --optuna-folds: {bad}. Valid range is 0..{NUM_FOLDS-1}")
    return folds

def create_config(args):
    if args.config == "unet3d":
        config = UNet3DConfig(fold=args.fold)
    elif args.config == "segresnet":
        config = SegResNetConfig(fold=args.fold)
    elif args.config == "unetr":
        config = UNETRConfig(fold=args.fold)
    elif args.config == "swinunetr":
        config = SwinUNETRConfig(fold=args.fold)
    else:
        raise ValueError(f"Unknown config: {args.config}")

    config.input_channels = 1
    return config


def create_model(args, config, device):
    if args.config == "unet3d":
        model = UNet3DModel(config)
    elif args.config == "segresnet":
        model = SegResNetModel(config)
    elif args.config == "unetr":
        model = UNETRModel(config)
    elif args.config == "swinunetr":
        model = SwinUNETRModel(config)
    else:
        raise ValueError(f"Unknown model type: {args.config}")
    return model.to(device)


def config_to_dict(config) -> Dict:
    cfg = {}
    for k, v in vars(config).items():
        if isinstance(v, (str, int, float, bool, list, tuple, dict, type(None))):
            cfg[k] = v
        else:
            cfg[k] = str(v)
    return cfg


def save_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    config,
    best_selection_score: float,
    val_metrics: Optional[Dict] = None,
):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": config_to_dict(config),
        "best_selection_score": float(best_selection_score),
        "val_metrics": {} if val_metrics is None else val_metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device):
    return torch.load(path, map_location=device)


def save_best_metrics_json(path: str, epoch: int, metrics: Dict):
    payload = {
        "epoch": int(epoch),
        **metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_val_metrics_csv(csv_path: str, epoch: int, metrics: Dict):
    fieldnames = [
        "epoch",
        "val_loss",
        "gtvp_mean_dsc",
        "gtvn_agg_dsc",
        "gtvn_f1agg",
        "task1_proxy_score",
        "gtvn_tp",
        "gtvn_fp",
        "gtvn_fn",
        "num_cases",
        "val_min_gtvn_size",
        "sw_overlap",
    ]

    row = {
        "epoch": int(epoch),
        "val_loss": metrics.get("val_loss", np.nan),
        "gtvp_mean_dsc": metrics.get("gtvp_mean_dsc", np.nan),
        "gtvn_agg_dsc": metrics.get("gtvn_agg_dsc", np.nan),
        "gtvn_f1agg": metrics.get("gtvn_f1agg", np.nan),
        "task1_proxy_score": metrics.get("task1_proxy_score", np.nan),
        "gtvn_tp": metrics.get("gtvn_tp", 0),
        "gtvn_fp": metrics.get("gtvn_fp", 0),
        "gtvn_fn": metrics.get("gtvn_fn", 0),
        "num_cases": metrics.get("num_cases", 0),
        "val_min_gtvn_size": metrics.get("val_min_gtvn_size", 0),
        "sw_overlap": metrics.get("sw_overlap", np.nan),
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_5fold_summary_csv(csv_path: str, all_results: List[Dict]):
    fieldnames = [
        "fold",
        "val_loss",
        "gtvp_mean_dsc",
        "gtvn_agg_dsc",
        "gtvn_f1agg",
        "task1_proxy_score",
        "gtvn_tp",
        "gtvn_fp",
        "gtvn_fn",
        "num_cases",
        "val_min_gtvn_size",
        "sw_overlap",
        "loss_name",
        "learning_rate",
        "weight_decay",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

        mean_row = {"fold": "mean"}
        for k in [
            "val_loss",
            "gtvp_mean_dsc",
            "gtvn_agg_dsc",
            "gtvn_f1agg",
            "task1_proxy_score",
            "gtvn_tp",
            "gtvn_fp",
            "gtvn_fn",
            "num_cases",
            "val_min_gtvn_size",
            "sw_overlap",
            "learning_rate",
            "weight_decay",
        ]:
            vals = [float(r[k]) for r in all_results if k in r]
            mean_row[k] = float(np.mean(vals)) if len(vals) > 0 else ""
        writer.writerow(mean_row)


def setup_trial_output_dirs(config, trial: Optional[optuna.trial.Trial]):
    """
    Put each Optuna trial into its own subdirectory so checkpoints/logs do not overwrite each other.
    """
    if trial is None:
        ensure_dir(config.fold_dir)
        ensure_dir(config.log_dir)
        ensure_dir(config.checkpoint_dir)
        return

    trial_name = f"trial_{trial.number:04d}"

    base_fold_dir = config.fold_dir
    base_log_dir = config.log_dir
    base_ckpt_dir = config.checkpoint_dir

    config.fold_dir = os.path.join(base_fold_dir, trial_name)
    config.log_dir = os.path.join(base_log_dir, trial_name)
    config.checkpoint_dir = os.path.join(base_ckpt_dir, trial_name)

    ensure_dir(config.fold_dir)
    ensure_dir(config.log_dir)
    ensure_dir(config.checkpoint_dir)


def suggest_hparams(trial: optuna.trial.Trial, args, config) -> Dict:
    """
    Search space for Optuna.
    Keep it small and high-value first.
    """
    loss_choices = args.optuna_losses
    if len(loss_choices) == 0:
        raise ValueError("--optuna-losses must contain at least one loss name.")
    
    config.learning_rate = trial.suggest_float("learning_rate", 2e-4, 8e-4, log=True)
    # config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    
    config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-5, log=True)
    # config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    min_gtvn_size = trial.suggest_int("min_gtvn_size", 8, 30)
    # min_gtvn_size = trial.suggest_int("min_gtvn_size", 3, 15)

    loss_name = trial.suggest_categorical("loss_name", loss_choices)
    sw_overlap = trial.suggest_float("sw_overlap", 0.25, 0.5)

    return {
        "loss_name": loss_name,
        "min_gtvn_size": min_gtvn_size,
        "sw_overlap": sw_overlap,
    }


def run_training(args, trial: Optional[optuna.trial.Trial] = None) -> Dict:
    config = create_config(args)

    if args.epochs is not None:
        config.num_epochs = args.epochs

    if args.device:
        config.device = args.device

    setup_trial_output_dirs(config, trial)

    logger = setup_logging(config.log_dir)
    if trial is None:
        logger.info(f"Starting training for fold {args.fold}...")
    else:
        logger.info(f"Starting Optuna trial {trial.number} for fold {args.fold}...")

    logger.info(f"Configuration: {config}")

    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        logger.info(f"Using {device}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        if config.device == "cuda":
            logger.warning("CUDA not available, falling back to CPU.")
        logger.info(f"Using device: {device}")

    trial_params = {
        "loss_name": "dice_focal",
        "min_gtvn_size": args.min_gtvn_size,
        "sw_overlap": 0.41,
    }

    if trial is not None:
        if args.resume:
            logger.warning("Resume checkpoint is ignored in Optuna mode.")
        trial_params = suggest_hparams(trial, args, config)
        logger.info(f"Optuna trial params: {trial_params}")
        logger.info(
            f"Optuna learning_rate={config.learning_rate:.6g}, "
            f"weight_decay={config.weight_decay:.6g}"
        )

    model = create_model(args, config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    train_loader, val_loader = get_dataloaders(config, fold=args.fold)
    logger.info(
        f"Data loaded for fold {args.fold}: "
        f"{len(train_loader)} train batches, {len(val_loader)} val batches"
    )

    criterion = get_loss_function(trial_params["loss_name"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=config.num_epochs,
        power=config.poly_lr_power,
    )

    use_tb = config.use_tensorboard
    if trial is not None and args.disable_tensorboard_in_optuna:
        use_tb = False
    writer = SummaryWriter(config.log_dir) if use_tb else None

    metrics_csv_path = os.path.join(config.fold_dir, "val_metrics_history.csv")

    start_epoch = 0
    best_selection_score = -1.0
    best_val_metrics = None

    if args.resume and trial is None:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_selection_score = checkpoint.get(
            "best_selection_score",
            checkpoint.get("best_task1_proxy_score", -1.0),
        )
        best_val_metrics = checkpoint.get("val_metrics", None)
        logger.info(
            f"Resumed from epoch {start_epoch}, "
            f"previous best selection score: {best_selection_score:.4f}"
        )

    validate_every = max(1, args.validate_every)

    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(f"Train - Loss: {train_loss:.4f}")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)

        should_validate = ((epoch + 1) % validate_every == 0) or ((epoch + 1) == config.num_epochs)
        val_metrics = None

        if should_validate:
            logger.info(
                "Running validation with sliding window inference "
                f"(Task 1 aligned metrics, min_gtvn_size={trial_params['min_gtvn_size']}, "
                f"sw_overlap={trial_params['sw_overlap']:.2f})..."
            )
            val_metrics = evaluate_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                config=config,
                use_sliding_window=True,
                min_gtvn_size=trial_params["min_gtvn_size"],
                sw_overlap=trial_params["sw_overlap"],
            )

            logger.info(
                "Val - "
                f"Loss: {val_metrics['val_loss']:.4f}, "
                f"GTVp mean DSC: {val_metrics['gtvp_mean_dsc']:.4f}, "
                f"GTVn agg DSC: {val_metrics['gtvn_agg_dsc']:.4f}, "
                f"GTVn F1agg: {val_metrics['gtvn_f1agg']:.4f}, "
                f"Task1Proxy: {val_metrics['task1_proxy_score']:.4f}, "
                f"TP/FP/FN: {val_metrics['gtvn_tp']}/{val_metrics['gtvn_fp']}/{val_metrics['gtvn_fn']}"
            )

            append_val_metrics_csv(metrics_csv_path, epoch + 1, val_metrics)

            if writer:
                writer.add_scalar("Loss/validation", val_metrics["val_loss"], epoch)
                writer.add_scalar("Task1/GTVp_mean_DSC", val_metrics["gtvp_mean_dsc"], epoch)
                writer.add_scalar("Task1/GTVn_agg_DSC", val_metrics["gtvn_agg_dsc"], epoch)
                writer.add_scalar("Task1/GTVn_F1agg", val_metrics["gtvn_f1agg"], epoch)
                writer.add_scalar("Task1/ProxyScore", val_metrics["task1_proxy_score"], epoch)

            if trial is not None:
                current_optuna_value = float(val_metrics[args.optuna_metric])
                trial.report(current_optuna_value, step=epoch)
                if trial.should_prune():
                    logger.info(
                        f"Trial {trial.number} pruned at epoch {epoch + 1} "
                        f"with {args.optuna_metric}={current_optuna_value:.6f}"
                    )
                    if writer:
                        writer.close()
                    raise optuna.TrialPruned()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Learning rate: {current_lr:.6f}")

        if writer:
            writer.add_scalar("Learning_Rate", current_lr, epoch)

        if should_validate and val_metrics is not None:
            if trial is not None:
                current_score = float(val_metrics[args.optuna_metric])
                score_name = args.optuna_metric
            else:
                # current_score = float(val_metrics["task1_proxy_score"])
                # score_name = "task1_proxy_score"

                current_score = float(val_metrics["gtvn_f1agg"])
                score_name = "gtvn_f1agg"

            if current_score > best_selection_score:
                best_selection_score = current_score
                best_val_metrics = copy.deepcopy(val_metrics)
                best_path = os.path.join(config.checkpoint_dir, "best_model.pth")

                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    best_selection_score=best_selection_score,
                    val_metrics=val_metrics,
                )

                best_metrics_json = os.path.join(config.checkpoint_dir, "best_model_metrics.json")
                save_best_metrics_json(best_metrics_json, epoch + 1, val_metrics)

                logger.info(
                    f"New best model saved with {score_name}: {best_selection_score:.4f}"
                )

        should_save_checkpoint = False
        if config.save_checkpoint_every > 0:
            should_save_checkpoint = ((epoch + 1) % config.save_checkpoint_every == 0)

        if should_save_checkpoint or ((epoch + 1) == config.num_epochs):
            last_model_path = os.path.join(config.checkpoint_dir, "last_model.pth")
            save_checkpoint(
                path=last_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                best_selection_score=best_selection_score,
                val_metrics=val_metrics,
            )
            logger.info(f"Saved last model checkpoint at epoch {epoch + 1}")

    logger.info("Training completed!")
    if writer:
        writer.close()

    if best_val_metrics is None:
        best_val_metrics = {
            "gtvp_mean_dsc": 0.0,
            "gtvn_agg_dsc": 0.0,
            "gtvn_f1agg": 0.0,
            "task1_proxy_score": -1.0,
            "gtvn_tp": 0,
            "gtvn_fp": 0,
            "gtvn_fn": 0,
            "num_cases": 0,
            "val_loss": float("inf"),
            "val_min_gtvn_size": int(trial_params["min_gtvn_size"]),
            "sw_overlap": float(trial_params["sw_overlap"]),
        }

    best_val_metrics["fold"] = int(args.fold)
    best_val_metrics["loss_name"] = trial_params["loss_name"]
    best_val_metrics["learning_rate"] = float(config.learning_rate)
    best_val_metrics["weight_decay"] = float(config.weight_decay)
    return best_val_metrics


# def objective(trial: optuna.trial.Trial, args) -> float:
#     metrics = run_training(args, trial=trial)

#     trial.set_user_attr("gtvn_fp", int(metrics.get("gtvn_fp", 0)))
#     trial.set_user_attr("gtvn_fn", int(metrics.get("gtvn_fn", 0)))
#     trial.set_user_attr("gtvn_tp", int(metrics.get("gtvn_tp", 0)))
#     trial.set_user_attr("gtvn_agg_dsc", float(metrics.get("gtvn_agg_dsc", 0.0)))
#     trial.set_user_attr("gtvp_mean_dsc", float(metrics.get("gtvp_mean_dsc", 0.0)))
#     trial.set_user_attr("task1_proxy_score", float(metrics.get("task1_proxy_score", -1.0)))

    # return float(metrics[args.optuna_metric])

#把 Optuna 的 objective 从“单 fold”改成“多 fold 平均”
def objective(trial: optuna.trial.Trial, args) -> float:
    optuna_folds = parse_fold_list(args.optuna_folds)

    fold_results = []

    for fold in optuna_folds:
        fold_args = copy.deepcopy(args)
        fold_args.fold = fold

        metrics = run_training(fold_args, trial=trial)
        fold_results.append(metrics)

    mean_metrics = {}
    metric_keys = [
        "val_loss",
        "gtvp_mean_dsc",
        "gtvn_agg_dsc",
        "gtvn_f1agg",
        "task1_proxy_score",
        "gtvn_tp",
        "gtvn_fp",
        "gtvn_fn",
        "num_cases",
        "val_min_gtvn_size",
        "sw_overlap",
        "learning_rate",
        "weight_decay",
    ]

    for k in metric_keys:
        vals = [float(m[k]) for m in fold_results if k in m]
        mean_metrics[k] = float(np.mean(vals)) if len(vals) > 0 else None

    # 记录 trial 的聚合结果，方便看 Optuna study
    trial.set_user_attr("optuna_folds", optuna_folds)
    for k, v in mean_metrics.items():
        if v is not None:
            trial.set_user_attr(f"mean_{k}", float(v))

    return float(mean_metrics[args.optuna_metric])


def run_all_folds(args):
    if args.optuna:
        raise ValueError("--run-all-folds and --optuna cannot be used together in this script.")
    if args.resume:
        raise ValueError("--run-all-folds and --resume cannot be used together in this script.")

    all_results = []

    print("\n========== STANDARD 5-FOLD TRAINING ==========")
    for fold in range(NUM_FOLDS):
        print(f"\n========== Running fold {fold} / {NUM_FOLDS - 1} ==========")

        fold_args = copy.deepcopy(args)
        fold_args.fold = fold

        metrics = run_training(fold_args, trial=None)
        all_results.append(metrics)

    print("\n========== 5-FOLD SUMMARY ==========")
    for m in all_results:
        print(
            f"Fold {m['fold']}: "
            f"val_loss={m['val_loss']:.4f}, "
            f"gtvp_mean_dsc={m['gtvp_mean_dsc']:.4f}, "
            f"gtvn_agg_dsc={m['gtvn_agg_dsc']:.4f}, "
            f"gtvn_f1agg={m['gtvn_f1agg']:.4f}, "
            f"task1_proxy_score={m['task1_proxy_score']:.4f}, "
            f"TP/FP/FN={m['gtvn_tp']}/{m['gtvn_fp']}/{m['gtvn_fn']}"
        )

    mean_metrics = {}
    mean_keys = [
        "val_loss",
        "gtvp_mean_dsc",
        "gtvn_agg_dsc",
        "gtvn_f1agg",
        "task1_proxy_score",
        "gtvn_tp",
        "gtvn_fp",
        "gtvn_fn",
        "num_cases",
        "val_min_gtvn_size",
        "sw_overlap",
        "learning_rate",
        "weight_decay",
    ]
    for k in mean_keys:
        vals = [float(m[k]) for m in all_results if k in m]
        mean_metrics[k] = float(np.mean(vals)) if len(vals) > 0 else None

    print("\nMean across 5 folds:")
    for k in ["val_loss", "gtvp_mean_dsc", "gtvn_agg_dsc", "gtvn_f1agg", "task1_proxy_score"]:
        print(f"  {k}: {mean_metrics[k]:.6f}")

    summary_dir = os.path.join("outputs", f"{args.config}_5fold_summary")
    os.makedirs(summary_dir, exist_ok=True)

    json_path = os.path.join(summary_dir, "summary.json")
    csv_path = os.path.join(summary_dir, "summary.csv")

    payload = {
        "config": args.config,
        "num_folds": NUM_FOLDS,
        "fold_results": all_results,
        "mean_metrics": mean_metrics,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    append_5fold_summary_csv(csv_path, all_results)

    print(f"\nSaved 5-fold summary JSON to: {json_path}")
    print(f"Saved 5-fold summary CSV to:  {csv_path}")


def main():
    args = parse_args()

    if args.run_all_folds:
        run_all_folds(args)
        return

    if args.optuna:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1,
        )

        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=bool(args.storage),
            direction="maximize",
            pruner=pruner,
        )

        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            timeout=args.optuna_timeout,
        )

        print("\nBest trial:")
        print(f"  value ({args.optuna_metric}): {study.best_trial.value:.6f}")
        print("  params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")

        if len(study.best_trial.user_attrs) > 0:
            print("  extra metrics:")
            for k, v in study.best_trial.user_attrs.items():
                print(f"    {k}: {v}")
    else:
        run_training(args, trial=None)


if __name__ == "__main__":
    main()