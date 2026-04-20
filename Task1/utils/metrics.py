"""
Task 1 evaluation metric utilities for HECKTOR 2025.

Main responsibilities:
- compute binary Dice for GTVp
- perform 3D connected component analysis
- compute lesion-level detection counts for GTVn (TP / FP / FN)
- compute lesion-aware aggregated DSC components for GTVn
- accumulate per-case validation statistics
- finalize Task 1 aligned validation metrics

"""

import numpy as np
from scipy.ndimage import label as cc_label
from typing import Dict, List, Tuple

VALID_LABELS = {0, 1, 2}
GTVN_IOU_THRESHOLD = 0.30


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


def update_task1_accumulators(acc: Dict, pred: np.ndarray, gt: np.ndarray):
    """Update Task 1 metrics accumulators from one case."""
    uniq_pred = set(np.unique(pred).tolist())
    uniq_gt = set(np.unique(gt).tolist())
    if not uniq_pred.issubset(VALID_LABELS):
        raise ValueError(f"Prediction has invalid labels: {sorted(uniq_pred)}")
    if not uniq_gt.issubset(VALID_LABELS):
        raise ValueError(f"Ground truth has invalid labels: {sorted(uniq_gt)}")

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