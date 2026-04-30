#!/usr/bin/env python3
"""
Main training entry script for HECKTOR 2025 Task 1 segmentation (PET-only).

This file serves as the high-level entry point for the training pipeline.
Its main responsibilities are:
- parse command-line arguments
- launch single-fold training
- launch standard 5-fold training
- launch Optuna hyperparameter search
- summarize multi-fold results

This script does not implement low-level training details directly.
Training/validation loop is delegated to `training.trainer.Trainer`, 
while metric computation, postprocessing, checkpoint handling, and training utilities are organized into separate
modules under `utils/` and `training/`.

train.py
  ↓
main()
  ↓
run_training(args)
  ↓
Trainer(args).run()
  ↓
create_config()
  ↓
create_model()
  ↓
if have SSL checkpoint: load SSL encoder
  ↓
get_dataloaders(fold=args.fold)
  ↓
for epoch:
    train_epoch()
    evaluate_epoch()
    save best_model / last_model
  ↓
return best_val_metrics

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

Goals
-----
- maximize gtvn_f1agg
- reduce gtvn_fp
- avoid excessive increase in gtvn_fn
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

from utils.metrics import (
    binary_dice,
    connected_components_3d,
    lesion_detection_counts,
    lesion_aware_aggregated_dsc_parts,
    update_task1_accumulators,
    finalize_task1_metrics,
)

from utils.postprocess import (
    check_values,
    remove_small_components,
    postprocess_prediction_task1,
)

from utils.checkpoint import (
    config_to_dict,
    save_checkpoint,
    load_checkpoint,
    save_best_metrics_json,
    append_val_metrics_csv,
    append_5fold_summary_csv,
)

from utils.train_utils import (
    NUM_FOLDS,
    parse_fold_list,
    create_config,
    create_model,
    ensure_dir,
    setup_trial_output_dirs,
    suggest_hparams,
)

from training.engine import evaluate_epoch, train_epoch
from training.trainer import Trainer

VALID_LABELS = {0, 1, 2}
GTVN_IOU_THRESHOLD = 0.30


def parse_args():
    parser = argparse.ArgumentParser(description="Train HECKTOR Task 1 segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="unet3d",
        choices=["unet3d", "ssl_unet_test", "segresnet", "unetr", "swinunetr"],
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

    parser.add_argument(
        "--ssl-pretrained-encoder",
        type=str,
        default=None,
        help="Path to pretrained white-box encoder checkpoint",
    )
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
        help="LR scale for pretrained encoder during SSL fine-tuning. Example: 0.1 means encoder_lr = base_lr * 0.1.",
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

def run_training(args, trial: Optional[optuna.trial.Trial] = None) -> Dict:
    trainer = Trainer(args=args, trial=trial)
    return trainer.run()

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