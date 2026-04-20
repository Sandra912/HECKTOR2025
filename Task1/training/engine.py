"""
Core epoch-level training and validation engine.

Main responsibilities:
- train the model for one epoch
- run validation for one epoch
- perform sliding window inference during validation
- apply prediction postprocessing
- compute Task 1 aligned validation metrics

"""
import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference

from utils.metrics import update_task1_accumulators, finalize_task1_metrics
from utils.postprocess import postprocess_prediction_task1


def evaluate_epoch(
    model,
    loader,
    criterion,
    device,
    config,
    use_sliding_window: bool = True,
    min_gtvn_size: int = 10,
    sw_overlap: float = 0.47613,
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
                pred_case = pred_np[i].astype("uint8")
                gt_case = label_np[i].astype("uint8")

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