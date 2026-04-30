"""Base configuration class for all models."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    
    # Data paths
    # data_root: str = "/data/HECKTOR2025/Task1"
    data_root: str = "/home/mi2488/hot/datasets/HECKTOR2025/Task1"

    train_images_dir: str = "imagesTr_resampled_npy"
    train_labels_dir: str = "labelsTr_resampled_npy"

    # train_images_dir: str = "imagesTr_train25_npy"
    # train_labels_dir: str = "labelsTr_train25_npy"

    # train_images_dir: str = "imagesTr_tune_npy"
    # train_labels_dir: str = "labelsTr_tune_npy"

    splits_file: str = "config/splits_available.json"
    
    # splits_file: str = "config/splits_train25.json"

    # splits_file: str = "config/splits_tune_train30_valexisting.json"
    
    # Data properties
    input_channels: int = 1  # CT + PET
    num_classes: int = 3     # background + primary tumor + metastatic tumor
    # spatial_size: Tuple[int, int, int] = (128, 128, 128) 
    spatial_size: Tuple[int, int, int] = (128, 128, 80)
    
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 5.38e-4
    weight_decay: float = 4.56e-5
    num_epochs: int = 300

    # Early stopping
    use_early_stopping: bool = False
    early_stop_patience: int = 20 
    early_stop_min_delta: float = 1e-4
    early_stop_metric: str = "gtvn_f1agg"
    
    # Scheduler parameters
    # PolyLR scheduler parameters
    poly_lr_power: float = 0.95
    poly_lr_min_lr: float = 1e-6
    
    # Data augmentation
    use_augmentation: bool = True
    aug_probability: float = 0.5
    rotation_range: float = 15.0
    scaling_range: float = 0.1
    translation_range: float = 10.0 # 平移
    
    # System parameters
    device: str = "cuda"
    num_workers: int = 48  # 用4个子进程加载数据
    pin_memory: bool = True #True
    
    # Data caching parameters
    cache_rate: float = 0.25  # Cache 25% of training data for faster loading
    
    # Checkpointing and logging
    save_checkpoint_every: int = 1 # Save checkpoint every n epochs
    use_tensorboard: bool = True
    
    # Output directories
    experiment_name: str = "baseline"
    output_dir: str = "/home/mi2488/hot/projects/HECKTOR2025/Task1/outputs"
    fold: int = 0
    
    def __post_init__(self):
        """Setup output directories with fold-specific structure."""
        # Create fold-specific directory structure
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.fold_dir = os.path.join(self.experiment_dir, f"fold_{self.fold}")
        self.checkpoint_dir = os.path.join(self.fold_dir, "checkpoints")
        self.log_dir = os.path.join(self.fold_dir, "logs")
        
        # Create directories
        for dir_path in [self.experiment_dir, self.fold_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
