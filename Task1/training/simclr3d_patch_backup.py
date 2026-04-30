"""
Local patch-level SSL training.

batch["view1"], batch["view2"]
→ model get local patch embeddings
→ local InfoNCE
→ optimizer update
→ save encoder
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.local_contrastive_loss import local_nt_xent_loss


def compute_local_pos_neg_margin(z1, z2):
    """
    z1, z2: [B, N, C]

    Positive:
        z1[b, n] 与 z2[b, n]

    Negative:
        z1[b, n] 与其他 z2 patch
    """
    B, N, C = z1.shape

    a = z1.reshape(B * N, C)
    b = z2.reshape(B * N, C)

    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    sim = torch.matmul(a, b.T)  # [BN, BN]

    positives = sim.diag()
    positive_mean = positives.mean().item()

    mask = torch.eye(B * N, device=z1.device).bool()
    negatives = sim[~mask]
    negative_mean = negatives.mean().item()

    margin = positive_mean - negative_mean

    return positive_mean, negative_mean, margin


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

        logging.basicConfig(
            filename=os.path.join(args.log_dir, "ssl_training.log"),
            level=logging.INFO,
        )

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        n_iter = 0

        logging.info(
            f"Start 3D local contrastive SSL training for {self.args.epochs} epochs"
        )

        for epoch in range(self.args.epochs):
            
            self.model.train()

            epoch_loss = 0.0
            epoch_pos = 0.0
            epoch_neg = 0.0
            epoch_margin = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"SSL Epoch {epoch + 1}/{self.args.epochs}",
            )

            for i, batch in enumerate(pbar):
                view1 = batch["view1"].to(self.args.device)
                view2 = batch["view2"].to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    # z1, z2: [B, N, C]
                    z1, patch_indices = self.model(view1)
                    z2, _ = self.model(view2, patch_indices=patch_indices)

                    loss = local_nt_xent_loss(
                        z1,
                        z2,
                        temperature=self.args.temperature,
                    )

                    pos_mean, neg_mean, margin = compute_local_pos_neg_margin(
                        z1,
                        z2,
                    )

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_value = loss.item()

                epoch_loss += loss_value
                epoch_pos += pos_mean
                epoch_neg += neg_mean
                epoch_margin += margin

                avg_loss = epoch_loss / (i + 1)
                avg_pos = epoch_pos / (i + 1)
                avg_neg = epoch_neg / (i + 1)
                avg_margin = epoch_margin / (i + 1)

                pbar.set_postfix(
                    # loss=f"{loss_value:.4f}",
                    avg_loss=f"{avg_loss:.4f}",
                    pos=f"{pos_mean:.4f}",
                    neg=f"{neg_mean:.4f}",
                    margin=f"{margin:.4f}",
                    # lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

                # if n_iter % self.args.log_every_n_steps == 0:
                #     self.writer.add_scalar("ssl/loss", loss_value, n_iter)
                #     self.writer.add_scalar("ssl/avg_loss", avg_loss, n_iter)
                #     self.writer.add_scalar(
                #         "ssl/lr",
                #         self.optimizer.param_groups[0]["lr"],
                #         n_iter,
                #     )
                #     self.writer.add_scalar("ssl/positive_mean", pos_mean, n_iter)
                #     self.writer.add_scalar("ssl/negative_mean", neg_mean, n_iter)
                #     self.writer.add_scalar("ssl/margin", margin, n_iter)

                #     # tqdm.write(
                #     #     f"[SSL-local] step={n_iter} "
                #     #     f"loss={loss_value:.4f} "
                #     #     f"avg_loss={avg_loss:.4f} "
                #     #     f"pos={pos_mean:.4f} "
                #     #     f"neg={neg_mean:.4f} "
                #     #     f"margin={margin:.4f}"
                #     # )

                n_iter += 1

            self.scheduler.step()

            mean_loss = epoch_loss / max(1, len(train_loader))
            mean_pos = epoch_pos / max(1, len(train_loader))
            mean_neg = epoch_neg / max(1, len(train_loader))
            mean_margin = epoch_margin / max(1, len(train_loader))

            if self.writer:
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

            ckpt_path = os.path.join(
                self.ckpt_dir,
                f"ssl_epoch_{epoch + 1:04d}.pth",
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "encoder_state_dict": self.model.encoder.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                ckpt_path,
            )

        final_path = os.path.join(self.ckpt_dir, "ssl_encoder_pretrained.pth")

        torch.save(
            {
                "encoder_state_dict": self.model.encoder.state_dict(),
            },
            final_path,
        )

        logging.info(f"Saved final encoder checkpoint to {final_path}")

        if self.writer:
            self.writer.close()

        return {
            "final_loss": float(mean_loss),
            "final_positive_mean": float(mean_pos),
            "final_negative_mean": float(mean_neg),
            "final_margin": float(mean_margin),
        }