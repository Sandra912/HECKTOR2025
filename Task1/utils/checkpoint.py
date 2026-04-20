"""
Checkpoint and result-saving utilities for training.

- converting config objects to serializable dictionaries
- saving and loading training checkpoints
- saving best validation metrics to JSON
- appending validation metrics to CSV history files
- writing 5-fold summary CSV files

"""
import os
import json
import csv
from typing import Dict, List, Optional

import numpy as np
import torch


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