from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random
from typing import Dict,List

def class_visualisation(galaxy_class: str, split="test", n=5):
    class_dir = Path("../data/Galaxy10_DECaLS") / split / galaxy_class

    if not class_dir.exists():
        raise ValueError(f"Class folder not found: {class_dir}")

    image_paths = list(class_dir.glob("*.jpg"))

    if len(image_paths) < n:
        raise ValueError(f"Not enough images in {galaxy_class}")

    selected_paths = random.sample(image_paths, n)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))

    for ax, img_path in zip(axes, selected_paths):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")

    plt.suptitle(f"Class: {' '.join(galaxy_class.split('_')).title()}")
    plt.tight_layout()
    plt.show()

def train_test_val_graph(galaxy_class: str):
    dataset_dir = Path("../data/Galaxy10_DECaLS")
    splits = ["train", "val", "test"]
    counts = {} 

    for split in splits:
        class_dir = dataset_dir / split / galaxy_class
        img_count = len(list(class_dir.glob("*.jpg")))
        counts[split] = img_count
    
    fig, ax = plt.subplots(figsize=(8, 5))

    heights = [counts[s] for s in splits]
    bars = ax.bar(splits, heights, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    ax.set_title(f"Distribution of Images: {' '.join(galaxy_class.split('_')).title()}", fontsize=14)
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Dataset Split")
    
    ax.bar_label(bars, padding=3)
    
    plt.show()

def plot_acc_loss_curves(results: Dict[str, List[float]], validation: bool) -> None:
    """
    Plots combined training and validation/test loss and accuracy curves.

    Args:
        results: Dictionary returned by the train() function.
        validation: If True, plots validation metrics.
                    If False, plots test metrics.
    """

    epochs = range(1, len(results["train_loss"]) + 1)

    eval_loss_key = "val_loss" if validation else "test_loss"
    eval_acc_key  = "val_acc"  if validation else "test_acc"
    eval_name     = "Validation" if validation else "Test"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, results["train_loss"], label="Train Loss")
    axes[0].plot(epochs, results[eval_loss_key], label=f"{eval_name} Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, results["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, results[eval_acc_key], label=f"{eval_name} Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
