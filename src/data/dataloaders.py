"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, List, Callable
from PIL import Image
import numpy as np
import torchvision.transforms
import torch

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

class ViTDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        class_names: List[str],
        image_size: int,
        patch_size: int,
        transform: torchvision.transforms
    ) -> None:
        
        self.root_dir = root_dir
        self.class_names = class_names
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        for idx, cls in enumerate(class_names):
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                self.samples.append(
                    (os.path.join(cls_path, img_name), idx)
                )

    def _patchify(self, img: torch.Tensor) -> torch.Tensor:
        """
        img shape: (C, H, W)
        returns: (num_patches, patch_dim)
        """
        c, h, w = img.shape
        p = self.patch_size

        img = img.unfold(1, p, p).unfold(2, p, p)
        img = img.contiguous().view(c, -1, p, p)
        img = img.permute(1, 0, 2, 3)
        patches = img.reshape(img.size(0), -1)

        return patches

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        patches = self._patchify(image)

        return patches, label


def dataloader_vit(
    data_dir: str,
    class_names: List[str],
    image_size: int,
    patch_size: int,
    batch_size: int,
    train_transform: torchvision.transforms,
    eval_transform: torchvision.transforms,
    validation: bool = True,
    seed: int = 42,
    num_workers: int = 4
):
    # Base dataset only to get indices
    base_dataset = ViTDataset(
        root_dir=data_dir,
        class_names=class_names,
        image_size=image_size,
        patch_size=patch_size,
        transform=eval_transform
    )

    dataset_size = len(base_dataset)
    indices = np.arange(dataset_size)

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(0.8 * dataset_size)
    val_end = int(0.9 * dataset_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_dataset = Subset(
        ViTDataset(
            data_dir, class_names, image_size, patch_size, train_transform
        ),
        train_indices
    )

    test_dataset = Subset(
        ViTDataset(
            data_dir, class_names, image_size, patch_size, eval_transform
        ),
        test_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    if validation:
        val_dataset = Subset(
            ViTDataset(
                data_dir, class_names, image_size, patch_size, eval_transform
            ),
            val_indices
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    return train_loader, test_loader


