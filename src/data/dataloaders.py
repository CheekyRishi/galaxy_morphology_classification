"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, List

DEFAULT_NUM_WORKERS = min(4, os.cpu_count())


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    """

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_data = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_data.classes

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
