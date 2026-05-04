"""
Image-level 3D SimCLR trainer.

batch["view1"], batch["view2"]
→ concatenate: [B, C, X, Y, Z] + [B, C, X, Y, Z] -> [2B, C, X, Y, Z]
→ model get global embedding: [2B, proj_dim]
→ InfoNCE
→ optimizer update
→ optional validation
→ save encoder
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SimCLR3D:
    def __init__(self, args, model, optimizer, scheduler):
        self.args = args
        self.model = model.to(args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        os.makedirs(args.log_dir, exist_ok=True)
        self.ckpt_dir = getattr(args, "ckpt_dir", args.log_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        logging.basicConfig(
            filename=os.path.join(args.log_dir, "ssl_training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def info_nce_loss(self, features):
        """
        Standard image-level SimCLR InfoNCE.

        features:
            [2B, D]
            first B are view1 embeddings
            second B are view2 embeddings

        Positive pairs:
            feature[i] <-> feature[i + B]
        """

        n_views = self.args.n_views
        batch_size = features.shape[0] // n_views

        labels = torch.cat(
            [torch.arange(batch_size) for _ in range(n_views)],
            dim=0,
        )
        labels = labels.to(self.args.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(
            labels.shape[0],
            dtype=torch.bool,
            device=self.args.device,
        )

        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0],
            -1,
        )

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0],
            -1,
        )

        positive_mean = positives.mean().item()
        negative_mean = negatives.mean().item()
        margin = positive_mean - negative_mean

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.args.temperature

        targets = torch.zeros(
            logits.shape[0],
            dtype=torch.long,
            device=self.args.device,
        )

        return logits, targets, positive_mean, negative_mean, margin

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """
        Run SSL validation on held-out AutoPET cases.

        Logs:
            ssl_val/loss
            ssl_val/positive_mean
            ssl_val/negative_mean
            ssl_val/margin
        """

        self.model.eval()

        val_loss_total = 0.0
        val_pos_total = 0.0
        val_neg_total = 0.0
        val_margin_total = 0.0
        num_batches = 0

        pbar = tqdm(
            val_loader,
            desc=f"SSL Val Epoch {epoch + 1}/{self.args.epochs}",
            dynamic_ncols=True,
            leave=False,
        )

        for batch in pbar:
            view1 = batch["view1"].to(self.args.device, non_blocking=True)
            view2 = batch["view2"].to(self.args.device, non_blocking=True)

            images = torch.cat([view1, view2], dim=0)

            with autocast(enabled=self.args.fp16_precision):
                features = self.model(images)
                logits, labels, pos_mean, neg_mean, margin = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

            loss_value = float(loss.item())

            val_loss_total += loss_value
            val_pos_total += float(pos_mean)
            val_neg_total += float(neg_mean)
            val_margin_total += float(margin)
            num_batches += 1

            pbar.set_postfix(
                val_loss=f"{loss_value:.4f}",
                pos=f"{pos_mean:.4f}",
                neg=f"{neg_mean:.4f}",
                margin=f"{margin:.4f}",
            )

        val_loss = val_loss_total / max(1, num_batches)
        val_pos = val_pos_total / max(1, num_batches)
        val_neg = val_neg_total / max(1, num_batches)
        val_margin = val_margin_total / max(1, num_batches)

        self.writer.add_scalar("ssl_val/loss", val_loss, epoch + 1)
        self.writer.add_scalar("ssl_val/positive_mean", val_pos, epoch + 1)
        self.writer.add_scalar("ssl_val/negative_mean", val_neg, epoch + 1)
        self.writer.add_scalar("ssl_val/margin", val_margin, epoch + 1)

        logging.info(
            f"Val Epoch {epoch + 1}: "
            f"loss={val_loss:.6f}, "
            f"positive_mean={val_pos:.6f}, "
            f"negative_mean={val_neg:.6f}, "
            f"margin={val_margin:.6f}"
        )

        print(
            f"[SSL Val] epoch={epoch + 1} "
            f"loss={val_loss:.4f} "
            f"pos={val_pos:.4f} "
            f"neg={val_neg:.4f} "
            f"margin={val_margin:.4f}"
        )

        self.model.train()

        return {
            "val_loss": val_loss,
            "val_positive_mean": val_pos,
            "val_negative_mean": val_neg,
            "val_margin": val_margin,
        }

    def train(self, train_loader, val_loader=None):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        n_iter = 0

        logging.info(f"Start 3D SimCLR training for {self.args.epochs} epochs")

        final_val_metrics = None

        best_val_margin = -float("inf")
        best_val_loss = float("inf")
        best_epoch = -1
        
        # early stop
        early_stop_counter = 0

        best_ssl_metric = getattr(self.args, "best_ssl_metric", "val_loss")
        early_stop_patience = getattr(self.args, "early_stop_patience", 30)
        early_stop_min_delta = getattr(self.args, "early_stop_min_delta", 1e-4)

        if best_ssl_metric == "val_loss":
            best_score = float("inf")      # loss 越小越好
        else:
            best_score = -float("inf")     # margin 越大越好
        
        last_epoch = 0
        for epoch in range(self.args.epochs):
            last_epoch = epoch + 1
            self.model.train()

            # best_val_margin = -float("inf")
            # best_val_loss = float("inf")
            # best_epoch = -1

            epoch_loss = 0.0
            epoch_pos = 0.0
            epoch_neg = 0.0
            epoch_margin = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"SSL Epoch {epoch + 1}/{self.args.epochs}",
                dynamic_ncols=True,
            )

            for i, batch in enumerate(pbar):
                view1 = batch["view1"].to(self.args.device, non_blocking=True)
                view2 = batch["view2"].to(self.args.device, non_blocking=True)

                images = torch.cat([view1, view2], dim=0)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)

                    # Debug once: should be [2B, proj_dim], e.g. [16, 128]
                    if epoch == 0 and i == 0:
                        print("features shape:", tuple(features.shape))

                    logits, labels, pos_mean, neg_mean, margin = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_value = float(loss.item())

                epoch_loss += loss_value
                epoch_pos += float(pos_mean)
                epoch_neg += float(neg_mean)
                epoch_margin += float(margin)

                avg_loss = epoch_loss / (i + 1)
                avg_pos = epoch_pos / (i + 1)
                avg_neg = epoch_neg / (i + 1)
                avg_margin = epoch_margin / (i + 1)

                pbar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg_loss=f"{avg_loss:.4f}",
                    pos=f"{pos_mean:.4f}",
                    neg=f"{neg_mean:.4f}",
                    margin=f"{margin:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

                if n_iter % self.args.log_every_n_steps == 0:
                    # self.writer.add_scalar("ssl/loss", loss_value, n_iter)
                    # self.writer.add_scalar("ssl/avg_loss", avg_loss, n_iter)
                    # self.writer.add_scalar("ssl/lr", self.optimizer.param_groups[0]["lr"], n_iter)
                    # self.writer.add_scalar("ssl/positive_mean", pos_mean, n_iter)
                    # self.writer.add_scalar("ssl/negative_mean", neg_mean, n_iter)
                    # self.writer.add_scalar("ssl/margin", margin, n_iter)

                    tqdm.write(
                        f"[SSL] step={n_iter} "
                        f"loss={loss_value:.4f} "
                        f"avg_loss={avg_loss:.4f} "
                        f"pos={pos_mean:.4f} "
                        f"neg={neg_mean:.4f} "
                        f"margin={margin:.4f}"
                    )

                n_iter += 1

            mean_loss = epoch_loss / max(1, len(train_loader))
            mean_pos = epoch_pos / max(1, len(train_loader))
            mean_neg = epoch_neg / max(1, len(train_loader))
            mean_margin = epoch_margin / max(1, len(train_loader))

            # Epoch-level train metrics, cleaner than step metrics
            self.writer.add_scalar("ssl_epoch/loss", mean_loss, epoch + 1)
            self.writer.add_scalar("ssl_epoch/positive_mean", mean_pos, epoch + 1)
            self.writer.add_scalar("ssl_epoch/negative_mean", mean_neg, epoch + 1)
            self.writer.add_scalar("ssl_epoch/margin", mean_margin, epoch + 1)
            self.writer.add_scalar(
                "ssl_epoch/lr",
                self.optimizer.param_groups[0]["lr"],
                epoch + 1,
            )

            logging.info(
                f"Epoch {epoch + 1}: "
                f"loss={mean_loss:.6f}, "
                f"positive_mean={mean_pos:.6f}, "
                f"negative_mean={mean_neg:.6f}, "
                f"margin={mean_margin:.6f}"
            )

            print(
                f"[SSL Train Epoch] epoch={epoch + 1} "
                f"loss={mean_loss:.4f} "
                f"pos={mean_pos:.4f} "
                f"neg={mean_neg:.4f} "
                f"margin={mean_margin:.4f}"
            )

            # Validation after each epoch
            # if val_loader is not None:
            #     final_val_metrics = self.validate(val_loader, epoch)

            # self.scheduler.step()

            # Validation after each epoch + save best pretrained encoder
            # if val_loader is not None:
            #     final_val_metrics = self.validate(val_loader, epoch)

            #     current_val_margin = float(final_val_metrics["val_margin"])
            #     current_val_loss = float(final_val_metrics["val_loss"])

            #     if current_val_margin > best_val_margin:
            #         best_val_margin = current_val_margin
            #         best_val_loss = current_val_loss
            #         best_epoch = epoch + 1

            #         best_path = os.path.join(self.ckpt_dir, "best_ssl_encoder_pretrained.pth")

            #         torch.save(
            #             {
            #                 "epoch": best_epoch,
            #                 "encoder_state_dict": self.model.encoder.state_dict(),
            #                 "model_state_dict": self.model.state_dict(),
            #                 "optimizer_state_dict": self.optimizer.state_dict(),
            #                 "best_val_margin": float(best_val_margin),
            #                 "best_val_loss": float(best_val_loss),
            #                 "train_loss": float(mean_loss),
            #                 "train_positive_mean": float(mean_pos),
            #                 "train_negative_mean": float(mean_neg),
            #                 "train_margin": float(mean_margin),
            #                 "val_metrics": final_val_metrics,
            #             },
            #             best_path,
            #         )

            #         print(
            #             f"Saved best SSL encoder to: {best_path} "
            #             f"(epoch={best_epoch}, val_margin={best_val_margin:.4f}, val_loss={best_val_loss:.4f})"
            #         )

            #         logging.info(
            #             f"Saved best SSL encoder at epoch {best_epoch}: "
            #             f"val_margin={best_val_margin:.6f}, val_loss={best_val_loss:.6f}"
            #         )
            if val_loader is not None:
                final_val_metrics = self.validate(val_loader, epoch)

                current_val_margin = float(final_val_metrics["val_margin"])
                current_val_loss = float(final_val_metrics["val_loss"])

                # 选择 early stop / best model 指标
                if best_ssl_metric == "val_loss":
                    current_score = current_val_loss
                    improved = current_score < (best_score - early_stop_min_delta)
                elif best_ssl_metric == "val_margin":
                    current_score = current_val_margin
                    improved = current_score > (best_score + early_stop_min_delta)
                else:
                    raise ValueError(f"Unknown best_ssl_metric: {best_ssl_metric}")

                if improved:
                    best_score = current_score
                    best_val_margin = current_val_margin
                    best_val_loss = current_val_loss
                    best_epoch = epoch + 1
                    early_stop_counter = 0

                    best_path = os.path.join(self.ckpt_dir, "best_ssl_encoder_pretrained.pth")

                    torch.save(
                        {
                            "epoch": best_epoch,
                            "encoder_state_dict": self.model.encoder.state_dict(),
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),

                            "best_ssl_metric": best_ssl_metric,
                            "best_score": float(best_score),
                            "best_val_margin": float(best_val_margin),
                            "best_val_loss": float(best_val_loss),

                            "train_loss": float(mean_loss),
                            "train_positive_mean": float(mean_pos),
                            "train_negative_mean": float(mean_neg),
                            "train_margin": float(mean_margin),
                            "val_metrics": final_val_metrics,
                        },
                        best_path,
                    )

                    print(
                        f"Saved best SSL encoder to: {best_path} "
                        f"(epoch={best_epoch}, metric={best_ssl_metric}, "
                        f"score={best_score:.6f}, val_margin={best_val_margin:.4f}, "
                        f"val_loss={best_val_loss:.4f})"
                    )

                    logging.info(
                        f"Saved best SSL encoder at epoch {best_epoch}: "
                        f"metric={best_ssl_metric}, "
                        f"score={best_score:.6f}, "
                        f"val_margin={best_val_margin:.6f}, "
                        f"val_loss={best_val_loss:.6f}"
                    )

                else:
                    early_stop_counter += 1

                    print(
                        f"No improvement in {best_ssl_metric}. "
                        f"Early stop counter: {early_stop_counter}/{early_stop_patience}"
                    )

                    logging.info(
                        f"No improvement in {best_ssl_metric}. "
                        f"Early stop counter: {early_stop_counter}/{early_stop_patience}"
                    )

                    if early_stop_counter >= early_stop_patience:
                        print(
                            f"Early stopping triggered at epoch {epoch + 1}. "
                            f"Best epoch={best_epoch}, best {best_ssl_metric}={best_score:.6f}"
                        )

                        logging.info(
                            f"Early stopping triggered at epoch {epoch + 1}. "
                            f"Best epoch={best_epoch}, best {best_ssl_metric}={best_score:.6f}"
                        )

                        break

            self.scheduler.step()

            # ckpt_path = os.path.join(self.ckpt_dir, f"ssl_epoch_{epoch + 1:04d}.pth")
            # torch.save(
            #     {
            #         "epoch": epoch + 1,
            #         "model_state_dict": self.model.state_dict(),
            #         "encoder_state_dict": self.model.encoder.state_dict(),
            #         "optimizer_state_dict": self.optimizer.state_dict(),
            #         "train_loss": float(mean_loss),
            #         "train_positive_mean": float(mean_pos),
            #         "train_negative_mean": float(mean_neg),
            #         "train_margin": float(mean_margin),
            #         "val_metrics": final_val_metrics,
            #     },
            #     ckpt_path,
            # )

        final_path = os.path.join(self.ckpt_dir, "ssl_encoder_pretrained.pth")

        torch.save(
            {
                # "epoch": int(self.args.epochs),
                # "epochs": int(self.args.epochs),
                "epoch": int(last_epoch),
                "epochs": int(last_epoch),
                "encoder_state_dict": self.model.encoder.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_ssl_metric": best_ssl_metric,
                "best_score": float(best_score),
                "final_loss": float(mean_loss),
                "final_positive_mean": float(mean_pos),
                "final_negative_mean": float(mean_neg),
                "final_margin": float(mean_margin),
                "final_val_metrics": final_val_metrics,
                "best_epoch": int(best_epoch),
                "best_val_margin": float(best_val_margin),
                "best_val_loss": float(best_val_loss),
            },
            final_path,
        )

        logging.info(f"Saved final encoder checkpoint to {final_path}")
        print(f"Saved final encoder checkpoint to: {final_path}")

        self.writer.close()

        result = {
            "final_loss": float(mean_loss),
            "final_positive_mean": float(mean_pos),
            "final_negative_mean": float(mean_neg),
            "final_margin": float(mean_margin),
            "best_ssl_metric": best_ssl_metric,
            "best_epoch": int(best_epoch),
            "best_val_margin": float(best_val_margin),
            "best_val_loss": float(best_val_loss),
        }

        if final_val_metrics is not None:
            result.update(final_val_metrics)

        return result