"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, List
from PIL import Image
import numpy as np

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
    def __init__(self, root_dir, class_names, image_size, patch_size):
        self.root_dir = root_dir
        self.class_names = class_names
        self.image_size = image_size
        self.patch_size = patch_size

        self.samples = []
        for idx, cls in enumerate(class_names):
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                self.samples.append(
                    (os.path.join(cls_path, img_name), idx)
                )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def _patchify(self, img):
        c, h, w = img.shape
        p = self.patch_size

        img = img.unfold(1, p, p).unfold(2, p, p)
        img = img.contiguous().view(c, -1, p, p)
        img = img.permute(1, 0, 2, 3)
        patches = img.reshape(img.size(0), -1)

        return patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        patches = self._patchify(image)

        return patches, label


def dataloader_vit(
    data_dir,
    class_names,
    image_size,
    patch_size,
    batch_size,
    validation=True,
    seed=42,
    num_workers=4
):
    """
    Returns train, val (optional), and test dataloaders.

    Split:
    - Train: 80%
    - Val:   10%
    - Test:  10%
    """

    dataset = ViTDataset(
        root_dir=data_dir,
        class_names=class_names,
        image_size=image_size,
        patch_size=patch_size
    )

    dataset_size = len(dataset)
    indices = np.arange(dataset_size)

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(0.8 * dataset_size)
    val_end = int(0.9 * dataset_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

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
        val_dataset = Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader

    return train_loader, test_loader


