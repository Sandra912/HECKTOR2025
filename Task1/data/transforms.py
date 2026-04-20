"""Data transforms for HECKTOR dataset (PET-only, clipped-to-SUV-like range)."""

import numpy as np
import torch
from monai.transforms import (
    Compose,
    MapTransform,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
    RandCropByLabelClassesd,
    SelectItemsd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    Lambdad,
    SpatialPadd,
)


class LoadNpzDictd(MapTransform):
    """
    Custom transform to load data from .npz files.
    Each .npz file is expected to contain:
      - 'image': numpy array
      - 'meta': metadata dict
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            filepath = d[key]
            loaded = np.load(filepath, allow_pickle=True)

            image_array = loaded["image"]
            meta_dict = loaded["meta"].item()

            d[key] = torch.from_numpy(image_array)
            d[f"{key}_meta_dict"] = meta_dict
        return d


def rename_pet_to_image(d):
    return {"image": d["pet"], "label": d["label"]}


PET_CLIP_MIN = 0.0
PET_CLIP_MAX = 30.0


def get_train_transforms(config):
    keys = ["pet", "label"]

    transforms = [
        LoadNpzDictd(keys=keys),

        Lambdad(
            keys=["pet"],
            func=lambda x: torch.clamp(x, min=PET_CLIP_MIN),
        ),

        ScaleIntensityRanged(
            keys=["pet"],
            a_min=PET_CLIP_MIN,
            a_max=PET_CLIP_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        # 先补到至少目标大小，再随机裁 patch
        SpatialPadd(
            keys=keys,
            spatial_size=config.spatial_size,
            method="end",
        ),

        RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            spatial_size=config.spatial_size,
            ratios=[0.1, 0.45, 0.45],
            num_classes=3,
            num_samples=3,
            allow_missing_keys=True,
            warn=False,
        ),
    ]

    if config.use_augmentation:
        transforms.extend([
            RandFlipd(
                keys=keys,
                spatial_axis=[0, 1, 2],
                prob=config.aug_probability,
            ),
            RandScaleIntensityd(
                keys=["pet"],
                factors=0.1,
                prob=config.aug_probability,
            ),
            RandShiftIntensityd(
                keys=["pet"],
                offsets=0.1,
                prob=config.aug_probability,
            ),
            RandGaussianNoised(
                keys=["pet"],
                std=0.01,
                prob=config.aug_probability,
            ),
            RandGaussianSmoothd(
                keys=["pet"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=config.aug_probability,
            ),
        ])

    transforms.extend([
        SelectItemsd(keys=["pet", "label"]),
        Lambdad(keys=["pet", "label"], func=lambda x: x),
        rename_pet_to_image,
        EnsureTyped(keys=["image", "label"]),
    ])

    return Compose(transforms)


# def get_validation_transforms(config):
#     keys = ["pet", "label"]

#     return Compose([
#         LoadNpzDictd(keys=keys),

#         Lambdad(
#             keys=["pet"],
#             func=lambda x: torch.clamp(x, min=PET_CLIP_MIN),
#         ),

#         ScaleIntensityRanged(
#             keys=["pet"],
#             a_min=PET_CLIP_MIN,
#             a_max=PET_CLIP_MAX,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
#         ),

#         CenterSpatialCropd(
#             keys=keys,
#             roi_size=config.spatial_size,
#         ),

#         # validation 也补齐到固定大小，避免 shape 不一致
#         SpatialPadd(
#             keys=keys,
#             spatial_size=config.spatial_size,
#             method="end",
#         ),

#         rename_pet_to_image,
#         EnsureTyped(keys=["image", "label"]),
#     ])
    
# 用整个体数据进行 validation
def get_validation_transforms(config):
    keys = ["pet", "label"]

    return Compose([
        LoadNpzDictd(keys=keys),

        Lambdad(
            keys=["pet"],
            func=lambda x: torch.clamp(x, min=PET_CLIP_MIN),
        ),

        ScaleIntensityRanged(
            keys=["pet"],
            a_min=PET_CLIP_MIN,
            a_max=PET_CLIP_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        rename_pet_to_image,
        EnsureTyped(keys=["image", "label"]),
    ])