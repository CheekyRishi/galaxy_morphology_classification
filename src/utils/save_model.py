from pathlib import Path
import torch


def save_model(
    model: torch.nn.Module,
    model_name: str,
    base_dir: str = "checkpoints"
):
    """
    Saves a PyTorch model to checkpoints/<architecture>/model_name.

    Folder is inferred from model_name, not class name.
    """

    assert model_name.endswith((".pt", ".pth")), \
        "model_name must end with .pt or .pth"

    name_lower = model_name.lower()

    if "resnet50" in name_lower:
        model_dir = "resnet50"
    elif "resnet26" in name_lower:
        model_dir = "resnet26"
    elif "vgg16" in name_lower:
        model_dir = "vgg16"
    elif "vgg19" in name_lower:
        model_dir = "vgg19"
    elif "vit" in name_lower:
        model_dir = "vit"
    else:
        model_dir = "custom_cnn"

    save_dir = Path("../" + base_dir) / model_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = save_dir / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
