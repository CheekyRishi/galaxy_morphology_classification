"""
Contains various utility functions for PyTorch model metrics' plotting and saving the model.
"""
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from typing import Dict,List


def save_model(
    model: torch.nn.Module,
    experiment_name: str,
    model_name: str,
    base_dir: str = "checkpoints"
):
    """
    Saves a PyTorch model to the appropriate checkpoints/<model_type>/ directory.

    Args:
        model: Trained PyTorch model.
        experiment_name: Name of the experiment (e.g. "exp_001_frozen").
        model_name: Filename for the saved model (must end with .pt or .pth).
        base_dir: Base directory for checkpoints (default: "checkpoints").
    """

    # Infer model type from class name
    raw_model_name = model.__class__.__name__.lower()

    # Normalize common architectures
    if "resnet" in raw_model_name:
        model_dir = "resnet50" if "50" in raw_model_name else raw_model_name
    elif "vgg" in raw_model_name:
        model_dir = raw_model_name
    elif "visiontransformer" in raw_model_name or "vit" in raw_model_name:
        model_dir = "vit"
    else:
        model_dir = "custom_cnn"

    # Create full save path
    save_dir = Path(base_dir) / model_dir / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Validate filename
    assert model_name.endswith(".pt") or model_name.endswith(".pth"), \
        "model_name must end with .pt or .pth"

    model_save_path = save_dir / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)


def plot_acc_loss_curves(results: Dict[str, List[float]], validation: bool) -> None:
    """
    Plots training and validation or testing loss and accuracy curves.

    Args:
        results: Dictionary returned by the train() function.
        validation: If True, plots validation metrics.
                    If False, plots test metrics.
    """

    epochs = range(1, len(results["train_loss"]) + 1)

    eval_loss_key = "val_loss" if validation else "test_loss"
    eval_acc_key  = "val_acc"  if validation else "test_acc"
    eval_name     = "Validation" if validation else "Test"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Train Loss
    axes[0, 0].plot(epochs, results["train_loss"])
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    # Validation/Test Loss
    axes[0, 1].plot(epochs, results[eval_loss_key])
    axes[0, 1].set_title(f"{eval_name} Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    # Train Accuracy
    axes[1, 0].plot(epochs, results["train_acc"])
    axes[1, 0].set_title("Train Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")

    # Validation/Test Accuracy
    axes[1, 1].plot(epochs, results[eval_acc_key])
    axes[1, 1].set_title(f"{eval_name} Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()
