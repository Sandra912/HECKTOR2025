"""
Load configuration
→ build SSL data augmentations
→ generate two views from the same PET image
→ perform SimCLR pretraining using SSLPretrainModel
→ save the encoder weights

Ideal behavior:
pos remains high, close to 1, meaning that different augmented views of the same case have very similar features. 

neg keeps decreasing.

margin keeps increasing, indicating an improved ability to separate the same case from different cases.
"""
#!/usr/bin/env python3
import os
import sys
import json
import copy
import argparse
import random

import numpy as np
import optuna
import torch
import torch.optim as optim

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UNet3DConfig
from models.ssl_pretrain_model import SSLPretrainModel
# from models.ssl_pretrain_model import LocalSSLPretrainModel
# from data.ssl_dataset import HECKTORSSLDataset
from data.ssl_dataset import AutoPETSSLDataset
from data.ssl_view_generator import ContrastiveLearningViewGenerator
from data.ssl_transforms import get_ssl_transform
from training.simclr3d import SimCLR3D


def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--n-views", type=int, default=2)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fp16-precision", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/mi2488/hot/projects/HECKTOR2025/Task1/outputs_ssl",
    )

    parser.add_argument(
    "--autopet-split-file",
    type=str,
    default="/home/mi2488/hot/projects/HECKTOR2025/Task1/config/autopet_fdg_bodycrop_ssl_split_80_20.json",
    )

    # debug
    parser.add_argument("--disable-debug", action="store_true")

    # optuna
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="hecktor_ssl")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--optuna-timeout", type=int, default=None)
    parser.add_argument(
        "--optuna-metric",
        type=str,
        default="final_loss",
        choices=["final_loss", "final_margin"],
    )

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# DEBUG
def check_view_alignment(v1, v2, thr=0.01):
    """
    v1, v2: [B, 1, X, Y, Z]
    Check whether the body masks of view1/view2 still overlap spatially.
    """
    with torch.no_grad():
        m1 = v1 > thr
        m2 = v2 > thr

        inter = (m1 & m2).sum(dim=(1, 2, 3, 4)).float()
        union = (m1 | m2).sum(dim=(1, 2, 3, 4)).float().clamp_min(1.0)

        iou = inter / union

        print("view body-mask IoU:", iou.detach().cpu().numpy())
        print("mean body-mask IoU:", float(iou.mean().item()))

def debug_ssl_batch(train_loader, model, device, max_print_samples=3):
    print("\n===== SSL sanity check =====")

    batch = next(iter(train_loader))
    v1 = batch["view1"]
    v2 = batch["view2"]

    print("view1 shape:", tuple(v1.shape))
    print("view2 shape:", tuple(v2.shape))

    global_diff = (v1 - v2).abs().mean().item()
    print(f"global mean abs diff: {global_diff:.6f}")

    for i in range(min(max_print_samples, v1.shape[0])):
        sample_diff = (v1[i] - v2[i]).abs().mean().item()
        print(f"sample {i} mean abs diff: {sample_diff:.6f}")

    print(
        f"view1 min/max/mean: {v1.min().item():.4f}, "
        f"{v1.max().item():.4f}, {v1.mean().item():.4f}"
    )
    
    #debug
    check_view_alignment(v1, v2, thr=0.01)

    print(
        f"view2 min/max/mean: {v2.min().item():.4f}, "
        f"{v2.max().item():.4f}, {v2.mean().item():.4f}"
    )

    model.eval()
    with torch.no_grad():
        z = model(v1.to(device))

        print("global embedding shape:", tuple(z.shape))

        sim = torch.matmul(z, z.T)

        print(
            f"embedding mean/std: {z.mean().item():.6f}, "
            f"{z.std().item():.6f}"
        )

        n_show = min(6, sim.shape[0])
        print(f"similarity matrix (top-left {n_show}x{n_show}):")
        print(sim[:n_show, :n_show].detach().cpu())

    model.train()
    print("===== End SSL sanity check =====\n")



def build_run_dirs(args):
    run_dir = os.path.join(args.output_dir, "autopet_fdg_80_20")
    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    return run_dir, log_dir, ckpt_dir

def run_ssl_training(args):
    config = UNet3DConfig()
    config.batch_size = args.batch_size

    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )


    args.device = device

    run_dir, log_dir, ckpt_dir = build_run_dirs(args)
    args.log_dir = log_dir
    args.ckpt_dir = ckpt_dir

    print(f"Using device: {device}")
    print(f"SSL output dir: {run_dir}")

    with open(os.path.join(run_dir, "ssl_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    base_transform = get_ssl_transform(config)
    view_transform = ContrastiveLearningViewGenerator(
        base_transform, n_views=args.n_views
    )

    # dataset = HECKTORSSLDataset(
    #     data_root=config.data_root,
    #     images_dir=config.train_images_dir,
    #     splits_file=config.splits_file,
    #     transform=view_transform,
    #     mode="all_folds",
    #     fold_idx=args.fold,
    # )

    train_dataset = AutoPETSSLDataset(
    split_file=args.autopet_split_file,
    split="train",
    transform=view_transform,
    )

    val_dataset = AutoPETSSLDataset(
        split_file=args.autopet_split_file,
        split="val",
        transform=view_transform,
    )

    print(f"AutoPET FDG SSL train size: {len(train_dataset)}")
    print(f"AutoPET FDG SSL val size: {len(val_dataset)}")

    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=(device.type == "cuda"),
    #     drop_last=True,
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )


    model = SSLPretrainModel(
        config=config,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
    )




    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    trainer = SimCLR3D(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    if not args.disable_debug:
        debug_ssl_batch(train_loader, model, device)

    # metrics = trainer.train(train_loader)
    metrics = trainer.train(train_loader, val_loader=val_loader)
    return metrics


def objective(trial, base_args):
    args = copy.deepcopy(base_args)

    # 基于你当前已经验证过更稳定的设置
    args.batch_size = 8
    args.n_views = 2
    args.proj_dim = 128

    # 搜索空间：先做第一轮稳定性搜索
    args.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    args.temperature = trial.suggest_categorical("temperature", [0.05, 0.07, 0.1, 0.2])
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])

    # 减少 trial 输出
    args.disable_debug = True

    # 每个 trial 单独目录
    args.output_dir = os.path.join(
        base_args.output_dir,
        "optuna",
        f"trial_{trial.number}"
    )

    metrics = run_ssl_training(args)

    trial.set_user_attr("final_loss", float(metrics["final_loss"]))
    trial.set_user_attr("final_margin", float(metrics["final_margin"]))
    trial.set_user_attr("final_positive_mean", float(metrics["final_positive_mean"]))
    trial.set_user_attr("final_negative_mean", float(metrics["final_negative_mean"]))

    if args.optuna_metric == "final_loss":
        return float(metrics["final_loss"])
    return float(metrics["final_margin"])


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.optuna:
        direction = "minimize" if args.optuna_metric == "final_loss" else "maximize"

        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=bool(args.storage),
            direction=direction,
        )

        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            timeout=args.optuna_timeout,
        )

        print("\nBest trial:")
        print(f"  value: {study.best_trial.value}")
        print("  params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")

        if len(study.best_trial.user_attrs) > 0:
            print("  extra metrics:")
            for k, v in study.best_trial.user_attrs.items():
                print(f"    {k}: {v}")
    else:
        metrics = run_ssl_training(args)
        print("\nFinal SSL metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")



if __name__ == "__main__":
    main()