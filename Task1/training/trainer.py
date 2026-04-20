"""
High-level training controller for HECKTOR 2025 Task 1.

Main responsibilities:
- initialize config, logger, model, optimizer, scheduler, and dataloaders
- handle resume / checkpoint restore logic
- manage epoch-by-epoch training and validation
- log metrics to TensorBoard
- save best and last checkpoints
- support both standard training and Optuna trial training

The Trainer acts as the central coordinator of the training process, while
lower-level metric, postprocessing, checkpoint, and engine logic are
delegated to separate utility modules.
"""
import os
import copy
from typing import Dict, Optional

import torch
import torch.optim as optim
import optuna
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloaders
from utils.losses import get_loss_function
from utils.logging import setup_logging
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_best_metrics_json,
    append_val_metrics_csv,
)
from utils.train_utils import (
    create_config,
    create_model,
    setup_trial_output_dirs,
    suggest_hparams,
)
from training.engine import evaluate_epoch, train_epoch


class Trainer:
    def __init__(self, args, trial: Optional[optuna.trial.Trial] = None):
        self.args = args
        self.trial = trial

    def run(self) -> Dict:
        args = self.args
        trial = self.trial

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
            "loss_name": "dice_ce",
            "min_gtvn_size": args.min_gtvn_size,
            "sw_overlap": 0.25,
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