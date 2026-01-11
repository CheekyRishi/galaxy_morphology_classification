from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random


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
    counts = {} # Use a different name from the loop variable

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