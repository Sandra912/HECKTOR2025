"""HECKTOR dataset implementation (PET-only)."""

import os
import glob
from typing import Dict, List, Optional, Callable
import torch
from torch.utils.data import Dataset


class HecktorDataset(Dataset):
    """HECKTOR dataset for 3D segmentation (PET-only)."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        case_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            images_dir: Directory containing PET .npz files
            labels_dir: Directory containing label .npz files
            transform: Data transforms to apply
            split: Dataset split ("train" or "val")
            case_ids: Optional list of specific case IDs to include
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.split = split

        if case_ids is not None:
            self.case_ids = case_ids
        else:
            self.case_ids = self._get_case_ids()

    def _get_case_ids(self) -> List[str]:
        """Get all case IDs from PET files."""
        pet_pattern = os.path.join(self.images_dir, "*_pet.npz")
        pet_files = glob.glob(pet_pattern)

        case_ids = []
        for pet_file in pet_files:
            case_id = os.path.basename(pet_file).replace("_pet.npz", "")
            case_ids.append(case_id)

        return sorted(case_ids)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        case_id = self.case_ids[idx]

        pet_path = os.path.join(self.images_dir, f"{case_id}_pet.npz")
        label_path = os.path.join(self.labels_dir, f"{case_id}_label.npz")

        data = {
            "pet": pet_path,
            "label": label_path,
            "case_id": case_id,
        }

        if self.transform:
            data = self.transform(data)

        return data