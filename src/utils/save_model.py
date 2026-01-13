"""
Contains various utility functions for PyTorch model metrics' plotting and saving the model.
"""
from pathlib import Path
import torch
from typing import Dict,List

def save_model(
    model: torch.nn.Module,
    model_name: str,
    base_dir: str = "checkpoints"
):
    """
    Saves a PyTorch model to the appropriate checkpoints/<model_type>/ directory.

    Args:
        model: Trained PyTorch model.
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
    save_dir = Path(base_dir) / model_dir 
    save_dir.mkdir(parents=True, exist_ok=True)

    # Validate filename
    assert model_name.endswith(".pt") or model_name.endswith(".pth"), \
        "model_name must end with .pt or .pth"

    model_save_path = save_dir / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)