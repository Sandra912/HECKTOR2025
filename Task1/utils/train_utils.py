"""
Main responsibilities:
- parse fold lists for multi-fold training or Optuna
- create config objects from command-line arguments
- create model instances from config selection
- create output directories
- prepare separate output folders for Optuna trials
- define and suggest Optuna hyperparameter search space

These utilities are shared by normal training, 5-fold training, and Optuna search.
"""
import os
from typing import Dict, List, Optional

import optuna

from config import UNet3DConfig, SegResNetConfig, UNETRConfig, SwinUNETRConfig
from models import UNet3DModel, SegResNetModel, UNETRModel, SwinUNETRModel

# from models.ssl_seg_model import Simple3DSegModel

NUM_FOLDS = 5


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
    # elif args.config == "ssl_unet_test":
    #     config = UNet3DConfig(fold=args.fold)
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


# def create_model(args, config, device):
#     if args.config == "unet3d":
#         model = UNet3DModel(config)
#     elif args.config == "segresnet":
#         model = SegResNetModel(config)
#     elif args.config == "unetr":
#         model = UNETRModel(config)
#     elif args.config == "swinunetr":
#         model = SwinUNETRModel(config)
#     else:
#         raise ValueError(f"Unknown model type: {args.config}")
#     return model.to(device)

def create_model(args, config, device):
    # if args.config == "ssl_unet_test":
    #     model = Simple3DSegModel(
    #         in_channels=config.input_channels,
    #         num_classes=config.num_classes,
    #         feature_channels=config.channels,
    #         dropout=config.dropout,
    #     )
    if args.config == "unet3d":
        model = UNet3DModel(config)
    elif args.config == "segresnet":
        model = SegResNetModel(config)
    elif args.config == "unetr":
        model = UNETRModel(config)
    elif args.config == "swinunetr":
        model = SwinUNETRModel(config)
    else:
        raise ValueError(f"Unsupported config: {args.config}")

    return model.to(device)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
    config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-5, log=True)

    min_gtvn_size = trial.suggest_int("min_gtvn_size", 8, 30)
    loss_name = trial.suggest_categorical("loss_name", loss_choices)
    sw_overlap = trial.suggest_float("sw_overlap", 0.25, 0.5)

    return {
        "loss_name": loss_name,
        "min_gtvn_size": min_gtvn_size,
        "sw_overlap": sw_overlap,
    }