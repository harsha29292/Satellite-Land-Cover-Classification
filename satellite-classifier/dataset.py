"""Dataset and dataloader utilities for EuroSAT."""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


def get_train_transforms():
    """Build and return training transforms."""
    data_cfg = CONFIG["data"]
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=data_cfg["hflip_p"]),
            transforms.RandomVerticalFlip(p=data_cfg["vflip_p"]),
            transforms.RandomRotation(degrees=data_cfg["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=data_cfg["brightness"],
                contrast=data_cfg["contrast"],
                saturation=data_cfg["saturation"],
            ),
            transforms.RandomResizedCrop(
                data_cfg["image_size"], scale=tuple(data_cfg["crop_scale"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ]
    )


def get_val_transforms():
    """Build and return validation transforms."""
    data_cfg = CONFIG["data"]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ]
    )


def get_datasets():
    """Create train and validation subsets from EuroSAT."""
    data_cfg = CONFIG["data"]
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_data = datasets.EuroSAT(
        root=data_cfg["root"],
        transform=train_transforms,
        download=data_cfg["download_train"],
    )
    val_data = datasets.EuroSAT(
        root=data_cfg["root"], transform=val_transforms, download=data_cfg["download_val"]
    )

    train_dataset, val_dataset = random_split(
        train_data,
        [data_cfg["train_split"], data_cfg["val_split"]],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    val_dataset.dataset = val_data
    return train_dataset, val_dataset


def get_dataloaders():
    """Create and return training and validation dataloaders."""
    data_cfg = CONFIG["data"]
    train_dataset, val_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=data_cfg["shuffle_train"],
        num_workers=data_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=data_cfg["shuffle_val"],
        num_workers=data_cfg["num_workers"],
    )
    return train_loader, val_loader
