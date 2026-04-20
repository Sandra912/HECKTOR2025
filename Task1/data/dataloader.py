from monai.data import DataLoader, CacheDataset, Dataset
from .transforms import get_train_transforms, get_validation_transforms
from typing import Tuple, List, Dict
import json
import os


def load_splits(splits_file: str) -> List[Dict[str, List[str]]]:
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    with open(splits_file, "r") as f:
        splits = json.load(f)
    return splits


def get_dataloaders(config, fold: int = 0) -> Tuple[DataLoader, DataLoader]:
    splits = load_splits(config.splits_file)
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} is out of range.")

    train_ids = splits[fold]["train"]
    val_ids = splits[fold]["val"]

    print(f"Using fold {fold}: {len(train_ids)} training cases, {len(val_ids)} validation cases")

    train_files = [
        {
            "pet": os.path.join(config.data_root, config.train_images_dir, f"{case_id}_pet.npz"),
            "label": os.path.join(config.data_root, config.train_labels_dir, f"{case_id}_label.npz"),
        }
        for case_id in train_ids
    ]

    val_files = [
        {
            "pet": os.path.join(config.data_root, config.train_images_dir, f"{case_id}_pet.npz"),
            "label": os.path.join(config.data_root, config.train_labels_dir, f"{case_id}_label.npz"),
        }
        for case_id in val_ids
    ]

    train_transforms = get_train_transforms(config)
    val_transforms = get_validation_transforms(config)

    use_cache = getattr(config, "cache_rate", 0.0) > 0.0
    loader_num_workers = int(getattr(config, "num_workers", 0))
    pin_memory = bool(getattr(config, "pin_memory", False))

    if use_cache:
        print(f"Creating CacheDataset for training with cache_rate={config.cache_rate}...")
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=config.cache_rate,
            num_workers=loader_num_workers,
        )
    else:
        print("Creating regular Dataset for training (cache disabled)...")
        train_ds = Dataset(
            data=train_files,
            transform=train_transforms,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(loader_num_workers > 0),
    )

    val_ds = Dataset(
        data=val_files,
        transform=val_transforms,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    return train_loader, val_loader