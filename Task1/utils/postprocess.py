"""
Prediction postprocessing utilities for HECKTOR 2025 Task 1.

Main responsibilities:
- check whether prediction labels are valid
- remove small connected components
- apply Task 1 postprocessing rules to predictions

Current Task 1 postprocessing includes:
- keep GTVp as label 1
- remove tiny predicted GTVn components
- enforce final label space to {0, 1, 2}

These functions are used during validation/inference postprocessing and are
kept separate from the training loop for cleaner project structure.
"""

import numpy as np
from utils.metrics import connected_components_3d

VALID_LABELS = {0, 1, 2}


def check_values(arr: np.ndarray, name: str):
    uniq = set(np.unique(arr).tolist())
    if not uniq.issubset(VALID_LABELS):
        raise ValueError(
            f"{name} has invalid labels: {sorted(uniq)}, expected subset of {sorted(VALID_LABELS)}"
        )


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


def postprocess_prediction_task1(
    pred: np.ndarray,
    min_gtvn_size: int = 10,
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