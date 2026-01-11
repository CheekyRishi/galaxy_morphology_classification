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
